#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>

static inline float min(float a, float b) {
    return (a < b) ? a : b;
}

static inline float max(float a, float b) {
    return (a > b) ? a : b;
}

// Data structures
typedef struct {
    long long wait_time;
    int *result;
    pthread_mutex_t *mutex;
    int *stop_flag;
} ThreadData;

typedef struct {
    int parent;
    int rank;
    int min_i, max_i, min_j, max_j;
} DisjointSetNode;

typedef struct {
    DisjointSetNode **sets;
    int n;
} HexDisjointSet;

typedef struct {
    float score;
    int best_i;
    int best_j;
} MoveResult;

typedef struct {
    int x, y;
    int dist;
} PQNode;


void* wait_until_timeout(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    if (data->wait_time > 1000) {
        fprintf(stderr, "Error: wait_time exceeds maximum allowed value.\n");
        return NULL;
    }
    sleep(data->wait_time);
    pthread_mutex_lock(data->mutex);
    *(data->stop_flag) = 1;
    pthread_mutex_unlock(data->mutex);
    return NULL;
}

void get_adjacent(int n, int i, int j, int adj[6][2], int is_max, int *count) {
    int directions[6][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, 1}, {1, -1}};
    *count = 0;
    for (int d = 0; d < 6; d++) {
        int ni = i + directions[d][0];
        int nj = j + directions[d][1];
        if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
            adj[*count][0] = ni;
            adj[*count][1] = nj;
            (*count)++;
        }
    }
}

// Disjoint Set
HexDisjointSet* create_hex_disjoint_set(int n) {
    HexDisjointSet* ds = malloc(sizeof(HexDisjointSet));
    ds->n = n;
    ds->sets = malloc(n * sizeof(DisjointSetNode*));
    for (int i = 0; i < n; i++) {
        ds->sets[i] = malloc(n * sizeof(DisjointSetNode));
        for (int j = 0; j < n; j++) {
            ds->sets[i][j].parent = i * n + j;
            ds->sets[i][j].rank = 0;
            ds->sets[i][j].min_i = ds->sets[i][j].max_i = i;
            ds->sets[i][j].min_j = ds->sets[i][j].max_j = j;
        }
    }
    return ds;
}

int find(HexDisjointSet* ds, int i, int j) {
    int id = i * ds->n + j;
    if (ds->sets[i][j].parent != id) {
        int parent_id = ds->sets[i][j].parent;
        int parent_i = parent_id / ds->n;
        int parent_j = parent_id % ds->n;
        ds->sets[i][j].parent = find(ds, parent_i, parent_j);
    }
    return ds->sets[i][j].parent;
}

void union_sets(HexDisjointSet* ds, int i1, int j1, int i2, int j2) {
    int root1 = find(ds, i1, j1);
    int root2 = find(ds, i2, j2);

    if (root1 != root2) {
        int root1_i = root1 / ds->n, root1_j = root1 % ds->n;
        int root2_i = root2 / ds->n, root2_j = root2 % ds->n;

        if (ds->sets[root1_i][root1_j].rank > ds->sets[root2_i][root2_j].rank) {
            ds->sets[root2_i][root2_j].parent = root1;
            // Update min/max values for root1
            ds->sets[root1_i][root1_j].min_i = min(min(ds->sets[root1_i][root1_j].min_i, 
                                                      ds->sets[root2_i][root2_j].min_i), 
                                                  min(i1, i2));
            ds->sets[root1_i][root1_j].max_i = max(max(ds->sets[root1_i][root1_j].max_i, 
                                                      ds->sets[root2_i][root2_j].max_i), 
                                                  max(i1, i2));
            ds->sets[root1_i][root1_j].min_j = min(min(ds->sets[root1_i][root1_j].min_j, 
                                                      ds->sets[root2_i][root2_j].min_j), 
                                                  min(j1, j2));
            ds->sets[root1_i][root1_j].max_j = max(max(ds->sets[root1_i][root1_j].max_j, 
                                                      ds->sets[root2_i][root2_j].max_j), 
                                                  max(j1, j2));
        } else {
            ds->sets[root1_i][root1_j].parent = root2;
            // Update min/max values for root2
            ds->sets[root2_i][root2_j].min_i = min(min(ds->sets[root1_i][root1_j].min_i, 
                                                      ds->sets[root2_i][root2_j].min_i), 
                                                  min(i1, i2));
            ds->sets[root2_i][root2_j].max_i = max(max(ds->sets[root1_i][root1_j].max_i, 
                                                      ds->sets[root2_i][root2_j].max_i), 
                                                  max(i1, i2));
            ds->sets[root2_i][root2_j].min_j = min(min(ds->sets[root1_i][root1_j].min_j, 
                                                      ds->sets[root2_i][root2_j].min_j), 
                                                  min(j1, j2));
            ds->sets[root2_i][root2_j].max_j = max(max(ds->sets[root1_i][root1_j].max_j, 
                                                      ds->sets[root2_i][root2_j].max_j), 
                                                  max(j1, j2));
            if (ds->sets[root1_i][root1_j].rank == ds->sets[root2_i][root2_j].rank) {
                ds->sets[root2_i][root2_j].rank++;
            }
        }
    }
}

void initialize_disjoint_sets(HexDisjointSet* ds1, HexDisjointSet* ds2, int **board, int n,int is_max) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == 1) {
                int adj[6][2], count;
                get_adjacent(n, i, j, adj,is_max, &count);
                for (int k = 0; k < count; k++) {
                    int ni = adj[k][0], nj = adj[k][1];
                    if (board[ni][nj] == 1) {
                        union_sets(ds1, i, j, ni, nj);
                    }
                }
            } else if (board[i][j] == 2) {
                int adj[6][2], count;
                get_adjacent(n, i, j, adj,is_max, &count);
                for (int k = 0; k < count; k++) {
                    int ni = adj[k][0], nj = adj[k][1];
                    if (board[ni][nj] == 2) {
                        union_sets(ds2, i, j, ni, nj);
                    }
                }
            }
        }
    }
}

int get_max_i_range(HexDisjointSet* ds) {
    int max_range = 0;
    for (int i = 0; i < ds->n; i++) {
        for (int j = 0; j < ds->n; j++) {
            if (ds->sets[i][j].parent == i * ds->n + j) { // Root of the tree
                int range = ds->sets[i][j].max_i - ds->sets[i][j].min_i;
                if (range > max_range) {
                    max_range = range;
                }
            }
        }
    }
    return max_range;
}

int get_max_j_range(HexDisjointSet* ds) {
    int max_range = 0;
    for (int i = 0; i < ds->n; i++) {
        for (int j = 0; j < ds->n; j++) {
            if (ds->sets[i][j].parent == i * ds->n + j) { // Root of the tree
                int range = ds->sets[i][j].max_j - ds->sets[i][j].min_j;
                if (range > max_range) {
                    max_range = range;
                }
            }
        }
    }
    return max_range;
}

void free_disjoint_sets(HexDisjointSet *ds1, HexDisjointSet *ds2) {
    if (ds1) {
        for (int i = 0; i < ds1->n; i++) {
            free(ds1->sets[i]);
        }
        free(ds1->sets);
        free(ds1);
    }

    if (ds2) {
        for (int i = 0; i < ds2->n; i++) {
            free(ds2->sets[i]);
        }
        free(ds2->sets);
        free(ds2);
    }
}

void update_disjoint_sets(HexDisjointSet *ds1, HexDisjointSet *ds2, int **board, int n, int i, int j, int player) {
    int adj[6][2], count;
    get_adjacent(n, i, j, adj, player == 1, &count);

    for (int k = 0; k < count; k++) {
        int ni = adj[k][0], nj = adj[k][1];
        if (board[ni][nj] == player) {
            if (player == 1) {
                union_sets(ds1, i, j, ni, nj);
            } else if (player == 2) {
                union_sets(ds2, i, j, ni, nj);
            }
        }
    }
}

static void copy_disjoint_sets(HexDisjointSet* dest, HexDisjointSet* src, int n) {
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            dest->sets[x][y] = src->sets[x][y];
        }
    }
}

// Disktra
void swap(PQNode *a, PQNode *b) {
    PQNode tmp = *a;
    *a = *b;
    *b = tmp;
}

void heapify_up(PQNode heap[], int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap[parent].dist <= heap[idx].dist) break;
        swap(&heap[parent], &heap[idx]);
        idx = parent;
    }
}

void heapify_down(PQNode heap[], int size, int idx) {
    while (1) {
        int smallest = idx;
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;

        if (left < size && heap[left].dist < heap[smallest].dist) smallest = left;
        if (right < size && heap[right].dist < heap[smallest].dist) smallest = right;

        if (smallest == idx) break;
        swap(&heap[idx], &heap[smallest]);
        idx = smallest;
    }
}

void push(PQNode heap[], int *size, int x, int y, int dist) {
    heap[*size] = (PQNode){x, y, dist};
    heapify_up(heap, (*size)++);
}

PQNode pop(PQNode heap[], int *size) {
    PQNode min = heap[0];
    heap[0] = heap[--(*size)];
    heapify_down(heap, *size, 0);
    return min;
}

// Determine if it is a winning move
int check_win_with_move(HexDisjointSet* ds1, HexDisjointSet* ds2, int player) {
    if (player == 1) {
        int max_j_range = get_max_j_range(ds1);
        if (max_j_range == ds1->n-1) {
            return 1;
        }
    } else if (player == 2) {
        int max_i_range = get_max_i_range(ds2);
        if (max_i_range == ds1->n-1) {
            return 1;
        }
    }
    return 0;
}

// Swap function for moves
void swap_moves(int moves[][2], int i, int j) {
    int temp_i = moves[i][0], temp_j = moves[i][1];
    moves[i][0] = moves[j][0];
    moves[i][1] = moves[j][1];
    moves[j][0] = temp_i;
    moves[j][1] = temp_j;
}

int partition_moves(int moves[][2], int low, int high, float **heuristics, int is_max) {
    int pivot_i = moves[high][0];
    int pivot_j = moves[high][1];
    float pivot_value = heuristics[pivot_i][pivot_j];

    int i = low - 1;
    for (int j = low; j < high; j++) {
        int current_i = moves[j][0];
        int current_j = moves[j][1];
        float current_value = heuristics[current_i][current_j];

        if ((is_max && current_value > pivot_value) || (!is_max && current_value < pivot_value)) {
            i++;
            swap_moves(moves, i, j);
        }
    }
    swap_moves(moves, i + 1, high);
    return i + 1;
}

// QuickSort function for sort the moves based on the heuristic table and whatever want to maximize or minimize
void quicksort_moves(int moves[][2], int low, int high, float **heuristics, int is_max) {
    if (low < high) {
        int pi = partition_moves(moves, low, high, heuristics, is_max);
        quicksort_moves(moves, low, pi - 1, heuristics, is_max);
        quicksort_moves(moves, pi + 1, high, heuristics, is_max);
    }
}

#define INF 100000

// Heuristic table calculation Function (basically calculates the distance from i,j to one side moving trow adjacents and with a cost determined for the adjacent type)
void calculate_heuristics(int n, int **board, int player, int side, float **heuristics, int **path_counts) {
    int **visited = malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        visited[i] = calloc(n, sizeof(int));
        for (int j = 0; j < n; j++) {
            heuristics[i][j] = INF;
        }
    }

    PQNode *heap = malloc(n * n * sizeof(PQNode));
    int heap_size = 0;

    // Initialize ghost node connected to the corresponding side
    if (side == 1) { // Top to bottom
        for (int j = 0; j < n; j++) {
            if (board[0][j] != 3 - player) {
                heuristics[0][j] = (board[0][j] == player) ? 0 : 1;
                path_counts[0][j] = 1;
                push(heap, &heap_size, 0, j, heuristics[0][j]);
            }
        }
    } else if (side == 2) { // Bottom to top
        for (int j = 0; j < n; j++) {
            if (board[n - 1][j] != 3 - player) {
                heuristics[n - 1][j] = (board[n - 1][j] == player) ? 0 : 1;
                path_counts[n - 1][j] = 1;
                push(heap, &heap_size, n - 1, j, heuristics[n - 1][j]);
            }
        }
    } else if (side == 3) { // Left to right
        for (int i = 0; i < n; i++) {
            if (board[i][0] != 3 - player) {
                heuristics[i][0] = (board[i][0] == player) ? 0 : 1;
                path_counts[i][0] = 1;
                push(heap, &heap_size, i, 0, heuristics[i][0]);
            }
        }
    } else if (side == 4) { // Right to left
        for (int i = 0; i < n; i++) {
            if (board[i][n - 1] != 3 - player) {
                heuristics[i][n - 1] = (board[i][n - 1] == player) ? 0 : 1;
                path_counts[i][n - 1] = 1;
                push(heap, &heap_size, i, n - 1, heuristics[i][n - 1]);
            }
        }
    }

    int dx[6] = {-1, 1, -1, 1, 0, 0};
    int dy[6] = {0, 0, 1, -1, -1, 1};

    while (heap_size > 0) {
        PQNode curr = pop(heap, &heap_size);
        int x = curr.x, y = curr.y;

        if (visited[x][y]) continue;
        visited[x][y] = 1;

        for (int d = 0; d < 6; d++) {
            int nx = x + dx[d];
            int ny = y + dy[d];

            if (nx >= 0 && ny >= 0 && nx < n && ny < n && board[nx][ny] != 3 - player) {
                int cost = (board[nx][ny] == player) ? 0 : 1;
                if (heuristics[x][y] + cost < heuristics[nx][ny]) {
                    path_counts[nx][ny] = path_counts[x][y];
                    heuristics[nx][ny] = heuristics[x][y] + cost;
                    push(heap, &heap_size, nx, ny, heuristics[nx][ny]);
                } else if (heuristics[x][y] + cost == heuristics[nx][ny]) {
                    path_counts[nx][ny] += path_counts[x][y];
                }
            }
        }
    }

    for (int i = 0; i < n; i++) free(visited[i]);
    free(visited);
    free(heap);
}

int count_blocked_positions(int **board, int n, int player, int i, int j) {
    int opponent = 3 - player;
    int blocked_count = 0;

    if (player == 1) {
        if (i > 0 && board[i - 1][j] == opponent) blocked_count+=2;
        else if (i < n - 1 && board[i + 1][j] == opponent) blocked_count+=2;
        else if (i > 0 && j < n - 1 && board[i - 1][j + 1] == opponent) blocked_count++;
        else if (i < n - 1 && j > 0 && board[i + 1][j - 1] == opponent) blocked_count++;

        if (i > 1 && j < n - 1 && board[i - 2][j + 1] == opponent) blocked_count++;
        if (i < n - 2 && j > 0 && board[i + 2][j - 1] == opponent) blocked_count++;

        if (i > 0 && j < n - 2 && board[i - 1][j + 2] == opponent) blocked_count++;
        if (i < n - 1 && j > 1 && board[i + 1][j - 2] == opponent) blocked_count++;

        if (i > 1 && board[i - 2][j] == opponent) blocked_count+=2;
        if (i < n - 2 && board[i + 2][j] == opponent) blocked_count+=2;
    }
    if (player == 2) {
        if (j > 0 && board[i][j - 1] == opponent) blocked_count+=2;
        else if (j < n - 1 && board[i][j + 1] == opponent) blocked_count+=2;
        else if (i > 0 && j < n - 1 && board[i - 1][j + 1] == opponent) blocked_count+=2;
        else if (i < n - 1 && j > 0 && board[i + 1][j - 1] == opponent) blocked_count+=2;

        if (i > 1 && j < n - 1 && board[i - 2][j + 1] == opponent) blocked_count++;
        if (i < n - 2 && j > 0 && board[i + 2][j - 1] == opponent) blocked_count++;

        if (i > 0 && j < n - 2 && board[i - 1][j + 2] == opponent) blocked_count++;
        if (i < n - 1 && j > 1 && board[i + 1][j - 2] == opponent) blocked_count++;

        if (j > 1 && board[i][j - 2] == opponent) blocked_count+=2;
        if (j < n - 2 && board[i][j + 2] == opponent) blocked_count+=2;
    }
    return blocked_count;
}
int evaluate_strategic_connection(int **board, int n, int player, int i, int j) {
    if (board[i][j] != 0) return -1;

    int opponent = 3 - player;
    int dir[6][2] = {
        {-1, 0}, {-1, 1}, {0, -1},
        {0, 1}, {1, -1}, {1, 0}
    };

    int score = 0;

    for (int d1 = 0; d1 < 6; d1++) {
        int i1 = i + dir[d1][0];
        int j1 = j + dir[d1][1];
        if (i1 < 0 || i1 >= n || j1 < 0 || j1 >= n) continue;
        if (board[i1][j1] != player) continue;

        for (int d2 = d1 + 1; d2 < 6; d2++) {
            int i2 = i + dir[d2][0];
            int j2 = j + dir[d2][1];
            if (i2 < 0 || i2 >= n || j2 < 0 || j2 >= n) continue;
            if (board[i2][j2] != player) continue;

            int blocked = 0;
            int supported = 0;

            for (int k = 0; k < 6; k++) {
                int ni = i1 + dir[k][0];
                int nj = j1 + dir[k][1];
                if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
                    if (board[ni][nj] == player) supported = 1;
                    if (board[ni][nj] == opponent) blocked = !blocked;
                }
            }

            for (int k = 0; k < 6; k++) {
                int ni = i2 + dir[k][0];
                int nj = j2 + dir[k][1];
                if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
                    if (board[ni][nj] == player) supported = 1;
                    if (board[ni][nj] == opponent) blocked = !blocked;
                }
            }

            if (supported) return 0;
            score += 1;
            if (blocked) score += 1;
        }
    }

    return score;
}

// Main depth-0 analysis function
MoveResult analyze_depth_zero(int **board, int n, int player, int is_max, 
                            HexDisjointSet* ds1, HexDisjointSet* ds2, 
                            float **heuristics, int *stop_flag) {
    MoveResult result = {is_max ? -1000000 : 1000000, -1, -1};

    if (*stop_flag) {
        return result;
    }

    // Iterate through all possible moves
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] != 0) continue;

            // Get heuristic value and make temporary move
            float h = heuristics[i][j];
            board[i][j] = is_max ? player : 3-player;

            // Create and copy temporary disjoint sets
            HexDisjointSet* temp_ds1 = create_hex_disjoint_set(n);
            HexDisjointSet* temp_ds2 = create_hex_disjoint_set(n);
            copy_disjoint_sets(temp_ds1, ds1, n);
            copy_disjoint_sets(temp_ds2, ds2, n);

            // Update connections
            update_disjoint_sets(temp_ds1, temp_ds2, board, n, i, j, player);

            // Undo move
            board[i][j] = 0;

            // Restore disjoint sets
            copy_disjoint_sets(ds1, temp_ds1, n);
            copy_disjoint_sets(ds2, temp_ds2, n);

            // Clean up temporary sets
            free_disjoint_sets(temp_ds1, temp_ds2);

            // Update best move if necessary
            if (is_max && h > result.score) {
                result.score = h;
                result.best_i = i;
                result.best_j = j;
            } else if (!is_max && h < result.score) {
                result.score = h;
                result.best_i = i;
                result.best_j = j;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        free(heuristics[i]);
    }
    free(heuristics);

    return result;
}


MoveResult minimax(int **board, int n, int depth, int alpha, int beta, int is_max, int player, int *stop_flag, HexDisjointSet* ds1, HexDisjointSet* ds2, int indent) {
    MoveResult result = {is_max ? -1000000 : 1000000, -1, -1};

    for (int i = 0; i < n && !(*stop_flag); i++) {
        for (int j = 0; j < n && !(*stop_flag); j++) {
            if (board[i][j] == 0) {

                board[i][j] = is_max ? player : 3 - player;

                HexDisjointSet* ds1_copy = create_hex_disjoint_set(n);
                HexDisjointSet* ds2_copy = create_hex_disjoint_set(n);

                for (int x = 0; x < n; x++) {
                    for (int y = 0; y < n; y++) {
                        ds1_copy->sets[x][y] = ds1->sets[x][y];
                        ds2_copy->sets[x][y] = ds2->sets[x][y];
                    }
                }
                update_disjoint_sets(ds1_copy, ds2_copy, board, n, i, j, player);

                if (check_win_with_move(ds1_copy, ds2_copy, is_max ? player : 3 - player)) {
                    result.score = is_max ? INF : -INF;
                    result.best_i = i;
                    result.best_j = j;
                    board[i][j] = 0;
                    return result;
                }

                free_disjoint_sets(ds1_copy, ds2_copy);

                board[i][j] = 0;
            }
        }
    }

    // Allocate memory for heuristics and path counts
    float **heuristics1 = malloc(n * sizeof(float *));
    float **heuristics2 = malloc(n * sizeof(float *));
    float **heuristics = malloc(n * sizeof(float *));
    int **path_counts1 = malloc(n * sizeof(int *));
    int **path_counts2 = malloc(n * sizeof(int *));
    
    for (int i = 0; i < n; i++) {
        heuristics1[i] = malloc(n * sizeof(float));
        heuristics2[i] = malloc(n * sizeof(float));
        heuristics[i] = malloc(n * sizeof(float));
        path_counts1[i] = malloc(n * sizeof(int));
        path_counts2[i] = malloc(n * sizeof(int));
    }

    // Calculate heuristics for both sides
    calculate_heuristics(n, board, player, (player == 1 ? 3 : 1), heuristics1, path_counts1);
    calculate_heuristics(n, board, player, (player == 1 ? 4 : 2), heuristics2, path_counts2);

    // Determine the maximum heuristic value (This heuristic determines the rest of n*n - the lenght of minimum pieces that needs to complete a path to win)
    float max_heuristic = -INF;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == 0) {
                float value = n * n - heuristics1[i][j] - heuristics2[i][j];
                if (value > max_heuristic) {
                    max_heuristic = value;
                }
            }
        }
    }

    // Determine the maximum blocked positions
    int max_blocked_positions = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == 0) {
                int blocked_positions = count_blocked_positions(board, n, player, i, j);
                if (blocked_positions > max_blocked_positions) {
                    max_blocked_positions = blocked_positions;
                }
            }
        }
    }

    // Determine the maximum strategic connection
    int max_strategic_connection = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == 0) {
                int strategic_connection = evaluate_strategic_connection(board, n, player, i, j);
                if (strategic_connection > max_strategic_connection) {
                    max_strategic_connection = strategic_connection;
                }
            }
        }
    }

    // Calculate final heuristics
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == 0) {
                board[i][j] = is_max ? player : 3 - player;

                HexDisjointSet* temp_ds1 = create_hex_disjoint_set(n);
                HexDisjointSet* temp_ds2 = create_hex_disjoint_set(n);
                for (int x = 0; x < n; x++) {
                    for (int y = 0; y < n; y++) {
                        temp_ds1->sets[x][y] = ds1->sets[x][y];
                        temp_ds2->sets[x][y] = ds2->sets[x][y];
                    }
                }

                int adj[6][2], count;
                get_adjacent(n, i, j, adj, is_max, &count);
                for (int k = 0; k < count; k++) {
                    int ni = adj[k][0], nj = adj[k][1];
                    if (board[ni][nj] == board[i][j]) {
                        if (board[i][j] == 1) {
                            union_sets(ds1, i, j, ni, nj);
                        } else {
                            union_sets(ds2, i, j, ni, nj);
                        }
                    }
                }

                if(player == 1){
                    int max_j_range = get_max_j_range(temp_ds1);
                    heuristics[i][j] = max_j_range*1
                        + 4 *(n * n - heuristics1[i][j] - heuristics2[i][j]) / max(1, max_heuristic)
                        + 6  * count_blocked_positions(board, n, player, i, j) / max(1, max_blocked_positions)
                        + 2 * evaluate_strategic_connection(board, n, player, i, j) / max(1, max_strategic_connection);
                }else{
                    int max_i_range = get_max_i_range(temp_ds2);
                    heuristics[i][j] = max_i_range*1
                        + 4 *(n * n - heuristics1[i][j] - heuristics2[i][j]) / max(1, max_heuristic)
                        + 6 * count_blocked_positions(board, n, player, i, j) / max(1, max_blocked_positions)
                        + 2 * evaluate_strategic_connection(board, n, player, i, j) / max(1, max_strategic_connection);
                }

                board[i][j] = 0;

                for (int x = 0; x < n; x++) {
                    for (int y = 0; y < n; y++) {
                        ds1->sets[x][y] = temp_ds1->sets[x][y];
                        ds2->sets[x][y] = temp_ds2->sets[x][y];
                    }
                }

                // Free temporary disjoint sets
                for (int x = 0; x < n; x++) {
                    free(temp_ds1->sets[x]);
                    free(temp_ds2->sets[x]);
                }
                free(temp_ds1->sets);
                free(temp_ds2->sets);
                free(temp_ds1);
                free(temp_ds2);

            } else {
                heuristics[i][j] = -INF;
            }
        }
    }

    // Free allocated memory for heuristics and path counts
    for (int i = 0; i < n; i++) {
        free(heuristics1[i]);
        free(heuristics2[i]);
        free(path_counts1[i]);
        free(path_counts2[i]);
    }
    free(heuristics1);
    free(heuristics2);
    free(path_counts1);
    free(path_counts2);

    
    
    // Depth 0 analisis:
    if (!(*stop_flag) && depth == 0) {
        return analyze_depth_zero(board,n,player,is_max,ds1,ds2,heuristics,stop_flag);
    }

    

    // Generate all possible moves
    int moves[n * n][2];
    int move_count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == 0) {
                moves[move_count][0] = i;
                moves[move_count][1] = j;
                move_count++;
            }
        }
    }

    // Sort moves using QuickSort
    quicksort_moves(moves, 0, move_count - 1, heuristics, is_max);
    


    // Explore moves in sorted order
    for (int m = 0; m < move_count && !(*stop_flag); m++) {
        int i = moves[m][0], j = moves[m][1];

        if(board[i][j] != 0) continue; 

        board[i][j] = is_max ? player : 3 - player;

        // Store the state of the disjoint sets before modification
        HexDisjointSet* temp_ds1 = create_hex_disjoint_set(n);
        HexDisjointSet* temp_ds2 = create_hex_disjoint_set(n);
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                temp_ds1->sets[x][y] = ds1->sets[x][y];
                temp_ds2->sets[x][y] = ds2->sets[x][y];
            }
        }

        int adj[6][2], count;
        get_adjacent(n, i, j, adj, is_max, &count);
        for (int k = 0; k < count; k++) {
            int ni = adj[k][0], nj = adj[k][1];
            if (board[ni][nj] == board[i][j]) {
                if (board[i][j] == 1) {
                    union_sets(ds1, i, j, ni, nj);
                } else {
                    union_sets(ds2, i, j, ni, nj);
                }
            }
        }


        // Continue exploring the move
        MoveResult curr = minimax(board, n, depth - 1, alpha, beta, !is_max, player, stop_flag, ds1, ds2, indent + 1);

        if(curr.best_i == -1 && curr.best_j == -1){
            continue;
        }

        board[i][j] = 0;

        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                ds1->sets[x][y] = temp_ds1->sets[x][y];
                ds2->sets[x][y] = temp_ds2->sets[x][y];
            }
        }

        // Free temporary disjoint sets
        for (int x = 0; x < n; x++) {
            free(temp_ds1->sets[x]);
            free(temp_ds2->sets[x]);
        }
        free(temp_ds1->sets);
        free(temp_ds2->sets);
        free(temp_ds1);
        free(temp_ds2);

        

        // Update the result based on the current move
        if (is_max && curr.score > result.score && curr.best_i != -1 && curr.best_j != -1) {
            result.score = curr.score;
            result.best_i = i;
            result.best_j = j;

            if(result.score>beta){
                break;
            }

            alpha = max(alpha,curr.score);
            
        } else if (!is_max && curr.score < result.score && curr.best_i != -1 && curr.best_j != -1) {
            result.score = curr.score;
            result.best_i = i;
            result.best_j = j;
            
            if(result.score<alpha){
                break;
            }

            beta = min(beta,result.score );
        }

    }


    for (int i = 0; i < n; i++) {
        free(heuristics[i]);
    }
    free(heuristics);

    return result;
}

float process_hex_board(int **board, int n, int depth, int player, long long wait_time, int *best_i, int *best_j) {
    pthread_t thread;
    int result = 0;
    int stop_flag = 0;
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    ThreadData data = {wait_time, &result, &mutex, &stop_flag};
    pthread_create(&thread, NULL, wait_until_timeout, &data);

    HexDisjointSet* ds1 = create_hex_disjoint_set(n);
    HexDisjointSet* ds2 = create_hex_disjoint_set(n);
    initialize_disjoint_sets(ds1, ds2, board, n,1);

    


    MoveResult best_move = minimax(board, n, depth, -100000, 100000, 1, player, &stop_flag, ds1, ds2,0);
    result = best_move.score;


    if (best_move.best_i != -1 && best_move.best_j != -1) {
        board[best_move.best_i][best_move.best_j] = player;
        *best_i = best_move.best_i;
        *best_j = best_move.best_j;
    }

    pthread_join(thread, NULL);
    pthread_mutex_destroy(&mutex);

    for (int i = 0; i < n; i++) {
        free(ds1->sets[i]);
        free(ds2->sets[i]);
    }
    free(ds1->sets);
    free(ds2->sets);
    free(ds1);
    free(ds2);


    
    return result;
}

