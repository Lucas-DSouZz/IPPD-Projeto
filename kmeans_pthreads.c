#define _POSIX_C_SOURCE 200112L     //  Necessário para CLOCK_MONOTONIC e pthread_barrier_t

#include <limits.h>                 //  Para LLONG_MAX
#include <pthread.h>                //  Biblioteca do Pthreads
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>                   //  Header correto para clock_gettime e struct timespec
#include <unistd.h>                 //  Para sysconf e _SC_NPROCESSORS_ONLN



//  --- Estruturas ---

//  Estrutura para representar um ponto no espaço D-dimensional
typedef struct {
    int cluster_id;     //  ID do cluster ao qual o ponto pertence
    int* coords;        //  Vetor de coordenadas inteiras
} Point;

//  Estrutura para enviar os argumentos para as threads
typedef struct {
    int thread_id, thread_count;            //  Dados das threads
    int M, D, K, I;                         //  Parâmetros globais
    int inicio, fim;                        //  Delimitadores da carga de trabalho da thread
    int *local_counts, *cluster_counts;     //  Contagens
    long long *local_sums, *cluster_sums;   //  Vetor de somas
    Point *points, *centroids;              //  Parâmetros globais
    pthread_barrier_t* barrier;
} Args;



//  --- Funções Utilitárias ---

/**
 *  @brief Calcula a distância Euclidiana ao quadrado entre dois pontos com coordenadas inteiras.
 *  Usa 'long long' para evitar overflow no cálculo da distância e da diferença.
 *  O 'static inline' sugere ao compilador para inserir o código diretamente no local da chamada, melhorando o desempenho.
 *  O 'restrict' informa ao compilador que os ponteiros não se sobrepõem, permitindo otimizações adicionais.
 *  @return A distância Euclidiana ao quadrado como um long long.
 */
static inline long long euclidean_dist_sq(const Point* restrict p1, const Point* restrict p2, int D) {
    long long dist = 0;
    for (int i = 0; i < D; i++) {
        long long diff = (long long)p1->coords[i] - p2->coords[i];
        dist += diff * diff;
    }
    return dist;
}

/**
 *  @brief Lê os dados de pontos (inteiros) de um arquivo de texto.
 */
void read_data_from_file(const char* filename, Point* points, int M, int D) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Erro: não foi possível abrir o arquivo '%s'.\n", filename);
        exit(EXIT_FAILURE);
    } 

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            if (fscanf(file, "%d", &points[i].coords[j]) != 1) {
                fprintf(stderr, "Erro: arquivo de dados mal formatado ou incompleto.\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);
}

/**
 *  @brief Inicializa os centroides escolhendo K pontos aleatórios do dataset.
 */
void initialize_centroids(Point* points, Point* centroids, int M, int D, int K) {
    srand(42);  //  Semente fixa para reprodutibilidade

    int* indices = (int*)malloc(M * sizeof(int));
    if (indices == NULL) {
        fprintf(stderr, "Erro: falha na alocação de memória.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < M; i++) {
        indices[i] = i;
    }

    for (int i = 0; i < M; i++) {
        int j = rand() % M;
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    for (int i = 0; i < K; i++) {
        memcpy(centroids[i].coords, points[indices[i]].coords, D * sizeof(int));
    }

    free(indices);
}



//  --- Funções Principais do K-Means ---

/**
 *  @brief Fase de Atribuição: Associa cada ponto ao cluster do centroide mais próximo.
 */
void assign_points_to_cluster(const Args* a) {
    //  Variáveis armazenadas localmente para otimização.
    int D = a->D;
    int K = a->K;
    Point* points = a->points;
    Point* centroids = a->centroids;

    for (int i = a->inicio; i < a->fim; i++) {     //  Delimitado pela carga de trabalho da thread.
        int best_cluster = -1;
        long long min_dist = LLONG_MAX;

        for (int j = 0; j < K; j++) {
            long long dist = euclidean_dist_sq(&points[i], &centroids[j], D);

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }

        points[i].cluster_id = best_cluster;
    }
}

/**
 *  @brief Fase de Atualização: Recalcula a posição de cada centroide como a média
 *  (usando divisão inteira) de todos os pontos atribuídos ao seu cluster.
 */
void update_centroids(const Args* a) {
    //  Variáveis armazenadas localmente para otimização.
    int D = a->D;
    int K = a->K;
    Point* points = a->points;
    Point* centroids = a->centroids;

    // Aqui, cada thread calcula suas somas e contagens locais, que serão combinadas posteriormente.
    int* restrict local_counts = &a->local_counts[(size_t)a->thread_id * K];
    long long* restrict local_sums = &a->local_sums[(size_t)a->thread_id * D * K];
    memset(local_counts, 0, sizeof(int) * K);
    memset(local_sums, 0, sizeof(long long) * D * K);

    for (int i = a->inicio; i < a->fim; i++) {
        int cluster_id = points[i].cluster_id;
        local_counts[cluster_id]++;

        for (int j = 0; j < D; j++) {
            local_sums[(size_t)cluster_id * D + j] += points[i].coords[j];
        }
    }

    pthread_barrier_wait(a->barrier);   //  Aguarda todas as threads chegarem aqui antes de combinar os resultados.

    if (a->thread_id == 0) {    //  Apenas a thread 0 combina os resultados, para evitar condições de corrida.
        int* restrict cluster_counts = a->cluster_counts;
        long long* restrict cluster_sums = a->cluster_sums;
        memset(cluster_counts, 0, sizeof(int) * K);
        memset(cluster_sums, 0, sizeof(long long) * D * K);

        //  Ponteiros auxiliares para facilitar o acesso aos dados locais e manter a indexação correta.
        int* aux_counts = a->local_counts;
        long long* aux_sums = a->local_sums;

        //  Combina os resultados locais de todas as threads.
        for (int t = 0; t < a->thread_count; t++) {
            int* t_count = &aux_counts[(size_t)t * K];
            long long* t_sum = &aux_sums[(size_t)t * D * K];

            for (int i = 0; i < K; i++) {
                cluster_counts[i] += t_count[i];
                for (int j = 0; j < D; j++) {
                    cluster_sums[(size_t)i * D + j] += t_sum[(size_t)i * D + j];
                }
            }
        }

        //  Atualiza os centroides com as médias calculadas.
        for (int i = 0; i < K; i++) {
            if (cluster_counts[i] > 0) {
                for (int j = 0; j < D; j++) {
                    centroids[i].coords[j] = (int)(cluster_sums[(size_t)i * D + j] / cluster_counts[i]);
                }
            }
        }
    }

    pthread_barrier_wait(a->barrier);   //  Aguarda a atualização de todos os centroides antes de prosseguir.
}



//  --- Funções de Inicialização das Threads ---

/**
 *  @brief Inicializa os argumentos utilizados pelas threads.
 *  Divide os dados de entrada entre as threads e associa os ponteiros necessários.
 */
void initialize_threads(
    Args* args, int thread_count,
    int M, int D, int K, int I,
    int* local_counts, int* cluster_counts,
    long long* local_sums, long long* cluster_sums,
    Point* points, Point* centroids,
    pthread_barrier_t* barrier
) {
    //  Calcula a carga de trabalho para cada thread, garantindo que todos os pontos sejam processados.
    int chunk = (M + thread_count - 1) / thread_count;

    for (int t = 0; t < thread_count; t++) {
        //  Atribui os argumentos globais.
        args[t].thread_id = t; 
        args[t].thread_count = thread_count;
        args[t].M = M;
        args[t].D = D;
        args[t].K = K;
        args[t].I = I;
        args[t].local_counts = local_counts;
        args[t].cluster_counts = cluster_counts;
        args[t].local_sums = local_sums;
        args[t].cluster_sums = cluster_sums;
        args[t].points = points;
        args[t].centroids = centroids;
        args[t].barrier = barrier;

        //  Atribui a carga de trabalho específica da thread.
        args[t].inicio = chunk * t;
        args[t].fim = chunk * (t + 1);
        if(args[t].fim > M) args[t].fim = M;
    }
}


/**
 *  @brief Função executada por cada thread para realizar o K-Means.
 *  Cada iteração realiza a atribuição de pontos e a atualização dos centroides.
 *  A sincronização das threads é feita usando barreiras dentro de update_centroids().
 */
void* run_threads(void* arg) {
    Args* a = (Args*)arg;   //  Recupera os argumentos da thread.

    for (int i = 0; i < a->I; i++) {
        assign_points_to_cluster(a);    //  Fase de atribuição
        update_centroids(a);            //  Fase de atualização
    }

    return NULL;
}



//  --- Funções de Impressão ---

/**
 *  @brief Imprime os resultados finais e o checksum (como long long).
 */
void print_results(Point* centroids, int K, int D) {
    printf("--- Centroides Finais ---\n");

    long long checksum = 0;
    for (int i = 0; i < K; i++) {
        printf("Centroide %d: [", i);

        for (int j = 0; j < D; j++) {
            printf("%d", centroids[i].coords[j]);
            if (j < D - 1) printf(", ");
            checksum += centroids[i].coords[j];
        }

        printf("]\n");
    }

    printf("\n--- Checksum ---\n");
    printf("%lld\n", checksum); //  %lld para long long int
}

/**
 *  @brief Calcula e imprime o tempo de execução e o checksum final.
 *  A saída é formatada para ser facilmente lida por scripts:
 *  Linha 1: Tempo de execução em segundos (double)
 *  Linha 2: Checksum final (long long)
 */
void print_time_and_checksum(Point* centroids, int D, int K, double exec_time) {
    long long checksum = 0;

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < D; j++) {
            checksum += centroids[i].coords[j];
        }
    }

    //  Saída formatada para o avaliador
    printf("%lf\n", exec_time);
    printf("%lld\n", checksum);
}



//  --- Função Principal ---

int main(int argc, char* argv[]) {
    //  Validação e leitura dos argumentos de linha de comando.
    if (argc != 6) {
        fprintf(stderr, "Uso: %s <arquivo_dados> <M_pontos> <D_dimensoes> <K_clusters> <I_iteracoes>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];      //  Nome do arquivo de dados.
    int M = atoi(argv[2]);               //  Número de pontos.
    int D = atoi(argv[3]);               //  Número de dimensões.
    int K = atoi(argv[4]);               //  Número de clusters.
    int I = atoi(argv[5]);               //  Número de iterações.

    //  Número de threads a partir da variável de ambiente NUM_THREADS (padrão 4).
    const char *env_threads = getenv("NUM_THREADS");
    int thread_count; 
    
    if (env_threads != NULL) {
        thread_count = atoi(env_threads);
    } else {
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        thread_count = (nprocs > 0) ? (int)nprocs : 4;
    }

    if (M <= 0 || D <= 0 || K <= 0 || I <= 0 || K > M || thread_count <= 0) {
        fprintf(stderr, "Erros nos parâmetros. Verifique se M, D, K, I, thread_count > 0 e K <= M.\n");
        return EXIT_FAILURE;
    }

    //  Alocação de memória e verificação.
    int* all_coords = (int*)malloc((size_t)(M + K) * D * sizeof(int));
    Point* points = (Point*)malloc(M * sizeof(Point));
    Point* centroids = (Point*)malloc(K * sizeof(Point));

    if (!all_coords || !points || !centroids) {
        fprintf(stderr, "Erro: falha na alocação de memória.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < M; i++) points[i].coords = &all_coords[(size_t)i * D];
    for (int i = 0; i < K; i++) centroids[i].coords = &all_coords[(size_t)(M + i) * D];

    //  Leitura dos dados e inicialização (fora do tempo de medição).
    read_data_from_file(filename, points, M, D);
    initialize_centroids(points, centroids, M, D, K);

    //  Inicialização das estruturas das threads (fora do tempo de medição).
    pthread_t threads[thread_count];
    Args args[thread_count];
    pthread_barrier_t barrier;

    int* local_counts = calloc((size_t)thread_count * K, sizeof(int));
    int* cluster_counts = calloc((size_t)K, sizeof(int));
    long long* local_sums = calloc((size_t)thread_count * D * K, sizeof(long long));
    long long* cluster_sums = calloc((size_t)D * K, sizeof(long long));

    if (!local_counts || !cluster_counts || !local_sums || !cluster_sums) {
        fprintf(stderr, "Erro: falha na alocação de memória.\n");
        free(all_coords);
        free(points);
        free(centroids);
        return EXIT_FAILURE;
    }

    pthread_barrier_init(&barrier, NULL, thread_count);

    initialize_threads(
        args, thread_count,
        M, D, K, I,
        local_counts, cluster_counts,
        local_sums, cluster_sums,
        points, centroids,
        &barrier
    );

    //  Medição de tempo do algoritmo principal.
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start); //  Inicia o cronômetro.

    //  Criação das threads e execução do K-Means.
    for (int t = 0; t < thread_count; t++) {
        if (pthread_create(&threads[t], NULL, run_threads, &args[t]) != 0) {    //  Fork
            fprintf(stderr, "Erro: falha ao criar a thread %d.\n", t);
            return EXIT_FAILURE;
        }
    }

    //  Espera todas as threads terminarem.
    for (int t = 0; t < thread_count; t++) {
        if (pthread_join(threads[t], NULL) != 0) {                              //  Join
            fprintf(stderr, "Erro: falha ao juntar a thread %d.\n", t);
            return EXIT_FAILURE;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);   //  Para o cronômetro.

    //  Calcula o tempo decorrido, em segundos.
    double time_taken = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

    //  Apresentação dos resultados.
    print_time_and_checksum(centroids, D, K, time_taken);

    //  Limpeza da memória alocada e da barreira.
    pthread_barrier_destroy(&barrier);
    free(all_coords);
    free(points);
    free(centroids);
    free(local_counts);
    free(cluster_counts);
    free(local_sums);
    free(cluster_sums);

    return EXIT_SUCCESS;
}