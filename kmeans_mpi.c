#define _POSIX_C_SOURCE 199309L  // Necessário para CLOCK_MONOTONIC
#include <limits.h>              // Para LLONG_MAX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // Header correto para clock_gettime e struct timespec
#include </usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h>

// Estrutura para representar um ponto no espaço D-dimensional
typedef struct {
  int* coords;     // Vetor de coordenadas inteiras
  int cluster_id;  // ID do cluster ao qual o ponto pertence
} Point;

int n_procs, my_rank, bin_length, start;

// --- Funções Utilitárias ---
/**
 * @brief Calcula a distância Euclidiana ao quadrado entre dois pontos com coordenadas inteiras.
 * Usa 'long long' para evitar overflow no cálculo da distância e da diferença.
 * @return A distância Euclidiana ao quadrado como um long long.
 */
long long euclidean_dist_sq(Point* p1, Point* p2, int D) {
  long long dist = 0;
  for (int i = 0; i < D; i++) {
    long long diff = (long long)p1->coords[i] - p2->coords[i];
    dist += diff * diff;
  }
  return dist;
}

// --- Funções Principais do K-Means ---

/**
 * @brief Lê os dados de pontos (inteiros) de um arquivo de texto.
 */
void read_data_from_file(const char* filename, Point* points, int M, int D) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Erro: Não foi possível abrir o arquivo '%s'\n", filename);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < D; j++) {
      if (fscanf(file, "%d", &points[i].coords[j]) != 1) {
        fprintf(stderr, "Erro: Arquivo de dados mal formatado ou incompleto.\n");
        fclose(file);
        exit(EXIT_FAILURE);
      }
    }
  }

  fclose(file);
}

/**
 * @brief Inicializa os centroides escolhendo K pontos aleatórios do dataset.
 */
void initialize_centroids(Point* points, Point* centroids, int M, int K, int D) {
  srand(42);  // Semente fixa para reprodutibilidade

  int* indices = (int*)malloc(M * sizeof(int));
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

/**
 * @brief Fase de Atribuição: Associa cada ponto ao cluster do centroide mais próximo.
 */
void assign_points_to_clusters(Point* points, Point* centroids, int M, int K, int D, int* best_cluster_list) {
  // trocar M por bin_length faz com que cada processo calcule apenas parte do vetor
  // * depois o resultado tem que ser concatenado
  for (int i = 0; i < bin_length; i++) {
    int calculated_index = start + i;
    long long min_dist = LLONG_MAX;
    int best_cluster = -1;

    for (int j = 0; j < K; j++) {
      long long dist = euclidean_dist_sq(&points[calculated_index], &centroids[j], D);
      if (dist < min_dist) {
        min_dist = dist;
        best_cluster = j;
      }
    }
    points[calculated_index].cluster_id = best_cluster;
    best_cluster_list[calculated_index] = best_cluster;
  }

  // combina os valores de todos os processos e distribui o resultado para todos os processos depois
  // * seria o mesmo que fazer reduce e broadcast
  // * note que usamos MPI_MAX porque o vetor vem zerado
  MPI_Allreduce(MPI_IN_PLACE, best_cluster_list, M, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  // atualiza points para todos os valores menos os que já foram calculados
  for (int i = 0; i < start; i++)
    points[i].cluster_id = best_cluster_list[i];

  for (int i = start + bin_length; i < M; i++)
    points[i].cluster_id = best_cluster_list[i];
}

/**
 * @brief Fase de Atualização: Recalcula a posição de cada centroide como a média
 * (usando divisão inteira) de todos os pontos atribuídos ao seu cluster.
 */
void update_centroids(Point* points, Point* centroids, int M, int K, int D, long long* cluster_sums, int* cluster_counts) {
  // estes vetores agora são declarados e alocados em main() e recebidos como parâmetros
  // desta forma só precisamos de uma alocação

  // long long* cluster_sums = (long long*)calloc(K * D, sizeof(long long));
  // int* cluster_counts = (int*)calloc(K, sizeof(int));

  for (int i = 0; i < bin_length; i++) {
    int calculated_index = start + i;
    int cluster_id = points[calculated_index].cluster_id;
    cluster_counts[cluster_id]++;   // vai precisar de reduce
    for (int j = 0; j < D; j++) {
      cluster_sums[cluster_id * D + j] += points[calculated_index].coords[j];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, cluster_sums, K*D, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, cluster_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  for (int i = 0; i < K; i++) {
    if (cluster_counts[i] > 0) {
      for (int j = 0; j < D; j++) {
        // Divisão inteira para manter os centroides em coordenadas discretas
        centroids[i].coords[j] = cluster_sums[i * D + j] / cluster_counts[i];
      }
    }
  }
}

/**
 * @brief Imprime os resultados finais e o checksum (como long long).
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
  printf("%lld\n", checksum);  // %lld para long long int
}

/**
 * @brief Calcula e imprime o tempo de execução e o checksum final.
 * A saída é formatada para ser facilmente lida por scripts:
 * Linha 1: Tempo de execução em segundos (double)
 * Linha 2: Checksum final (long long)
 */
void print_time_and_checksum(Point* centroids, int K, int D, double exec_time) {
  long long checksum = 0;
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < D; j++) {
      checksum += centroids[i].coords[j];
    }
  }
  // Saída formatada para o avaliador
  printf("%lf\n", exec_time);
  printf("%lld\n", checksum);
}

// --- Função Principal ---

int main(int argc, char* argv[]) {
  // Validação e leitura dos argumentos de linha de comando
  if (argc != 6) {
    fprintf(stderr, "Uso: %s <arquivo_dados> <M_pontos> <D_dimensoes> <K_clusters> <I_iteracoes>\n", argv[0]);
    return EXIT_FAILURE;
  }

  MPI_Init(&argc, &argv);

  const char* filename = argv[1];  // Nome do arquivo de dados
  const int M = atoi(argv[2]);     // Número de pontos
  const int D = atoi(argv[3]);     // Número de dimensões
  const int K = atoi(argv[4]);     // Número de clusters
  const int I = atoi(argv[5]);     // Número de iterações

  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (M <= 0 || D <= 0 || K <= 0 || I <= 0 || K > M) {
    fprintf(stderr, "Erro nos parâmetros. Verifique se M,D,K,I > 0 e K <= M.\n");
    return EXIT_FAILURE;
  }

  // --- Alocação de Memória ---
  int* all_coords = (int*)malloc((M + K) * D * sizeof(int));
  Point* points = (Point*)malloc(M * sizeof(Point));
  Point* centroids = (Point*)malloc(K * sizeof(Point));

  // ... (verificação de alocação) ...
  for (int i = 0; i < M; i++) {
    points[i].coords = &all_coords[i * D];
  }

  for (int i = 0; i < K; i++) {
    centroids[i].coords = &all_coords[(M + i) * D];
  }

  // --- Preparação para a divisão de trabalho ---
  int base = M / n_procs;
  int remainder  = M % n_procs;

  // divide o resto entre os bins
  if (my_rank < remainder) {
      bin_length = base + 1;
      start = my_rank * bin_length;
  } else {
      bin_length = base;
      // pula os bins que tem itens extra adicionados do resto
      // e depois pula os bins que não tem itens adicionados
      start = remainder * (base + 1) + (my_rank - remainder) * base;
  }

  // como não podemos criar um MPI Datatype de Points, precisamos criar uma lista (best_cluster_list)
  // auxiliar a ser compartilhada entre os ranks, e vamos alocar os vetores fora dos loops
  // para evitar realocações desnecessárias durante as iterações
  int* best_cluster_list = (int*)calloc(M, sizeof(int));
  long long* cluster_sums = (long long*)calloc(K * D, sizeof(long long));
  int* cluster_counts = (int*)calloc(K, sizeof(int));

  // --- Preparação (Fora da medição de tempo) ---
  read_data_from_file(filename, points, M, D);
  initialize_centroids(points, centroids, M, K, D);

  // --- Medição de Tempo do Algoritmo Principal ---
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);  // Inicia o cronômetro

  // Laço principal do K-Means (A única parte que será medida)
  for (int iter = 0; iter < I; iter++) {
    // zera os vetores para uma nova rodada de cálculos
    memset(best_cluster_list, 0, M * sizeof(int));
    memset(cluster_sums, 0, K * D * sizeof(long long));
    memset(cluster_counts, 0, K * sizeof(int));

    assign_points_to_clusters(points, centroids, M, K, D, best_cluster_list);
    update_centroids(points, centroids, M, K, D, cluster_sums, cluster_counts);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);  // Para o cronômetro

  // Calcula o tempo decorrido em segundos
  double time_taken = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

  if (my_rank == 0)
    // --- Apresentação dos Resultados ---
    print_time_and_checksum(centroids, K, D, time_taken);

  // --- Limpeza ---
  free(best_cluster_list);
  free(cluster_sums);
  free(cluster_counts);
  free(all_coords);
  free(points);
  free(centroids);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
