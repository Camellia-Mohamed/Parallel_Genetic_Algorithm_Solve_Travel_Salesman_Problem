#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <random>
#include <omp.h>

using namespace std;

#include "data.h" 
vector<double> fitness; // total distance of each route
vector<int> best_route; // best route
double best_fitness = 1e9; // best fitness
omp_lock_t best_lock; // lock for best route


double distance(const City &a, const City &b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// Calculate total route distance
double calculateTotalDistance(const vector<int> &route)
{
    double dist = 0;
    for (int i = 0; i < CITY_COUNT - 1; ++i)
    {
            dist += distance(cities[route[i]], cities[route[i + 1]]);
    }
    dist += distance(cities[route[CITY_COUNT - 1]], cities[route[0]]);

    return dist;
}

// Dynamic mutation rate that increases when stuck
float getDynamicMutationRate(int generation, double current_best, double previous_best)
{
    static int fixed = 0;
    if (current_best < previous_best)
    {
        return MUTATION_RATE;
    }
    fixed++;
    return min(0.5, MUTATION_RATE * (1.0 + fixed / 1000.0));
}


int selectParent(mt19937 &rnd)
{
    uniform_int_distribution<> dist(0, POP_SIZE - 1);
    int best = dist(rnd);
    for (int i = 1; i < SELECTION_SIZE; ++i)
    {
        int idx = dist(rnd); // random root index
        if (fitness[idx] < fitness[best])
            best = idx;
    }
    return best;
}

// Order Crossover (OX) with random crossover points
vector<int> crossover(const vector<int> &parent1, const vector<int> &parent2, mt19937 &rnd)
{
    vector<int> child(CITY_COUNT, -1);
    vector<bool> used(CITY_COUNT, false);

    uniform_int_distribution<> dist(0, CITY_COUNT - 1);
    int start = dist(rnd);
    int end = dist(rnd);
    if (start > end)
        swap(start, end);
    // Copy the segment from parent1
    for (int i = start; i <= end ; ++i)
    {
        if (parent1[i] >= 0 && parent1[i] < CITY_COUNT)
            child[i] = parent1[i],
            used[parent1[i]] = true;
    }

    // Fill the rest from parent2
    int j = 0;
    for (int i = end + 1; i < CITY_COUNT; ++i)
    {
        while (j < CITY_COUNT && used[parent2[j]])
            ++j;
        if(j < CITY_COUNT){
            child[i] = parent2[j];
            used[parent2[j]] = true;
            ++j;
        }
    }
    for (int i = 0; i < CITY_COUNT; ++i)
    {
        if (child[i] < 0 || child[i] >= CITY_COUNT)
            return parent1;
    }
    return child;
}

// swap cities in route with probability mutation_rate
void mutate(vector<int> &route, mt19937 &rnd, float mutation_rate)
{
    uniform_real_distribution<> prob(0.0, 1.0);
    if (prob(rnd) < mutation_rate)
    {
        uniform_int_distribution<> dist(0, CITY_COUNT - 1);
        // Perform multiple swaps based on mutation rate
        int num_swaps = max(1, int(mutation_rate * 5));
        for (int s = 0; s < num_swaps; ++s)
        {
            int i = dist(rnd);
            int j = dist(rnd);
            swap(route[i], route[j]);
        }
    }
}

// Evaluate fitness of entire population and update best route
void evaluateFitness()
{
    if (fitness.size() != POP_SIZE)
        fitness.resize(POP_SIZE);

    double local_best_fitness = 1e9;
    vector<int> local_best_route(CITY_COUNT);
    vector<double> local_fitness(POP_SIZE);
    // set parallel region
    #pragma omp parallel
    {
        // split the iterations among threads with nowait to avoid synchronization
        #pragma omp for nowait
        for (int i = 0; i < POP_SIZE; ++i)
        {
            
            local_fitness[i] = calculateTotalDistance(population[i]);
            if (local_fitness[i] < local_best_fitness)
            {
                #pragma omp critical
                {
                    if (local_fitness[i] < local_best_fitness)
                    {
                        local_best_fitness = local_fitness[i];
                        local_best_route = population[i];
                    }
                }
            }
            
        }
    }

    // Update global fitness values
    #pragma omp parallel for
    for (int i = 0; i < POP_SIZE; ++i)
    {
        fitness[i] = local_fitness[i];
    }

    // Update global best solution
    #pragma omp critical
    if (local_best_fitness < best_fitness)
    {
        best_fitness = local_best_fitness;
        best_route = local_best_route;
    }
}

// Generate next generation
void nextGeneration()
{
    if (population.size() != POP_SIZE)
        return;

    vector<vector<int>> new_population(POP_SIZE);
    vector<mt19937> generators(omp_get_max_threads());

    // Initialize random number generators for each thread
    for (int i = 0; i < generators.size(); ++i)
        generators[i].seed(i + time(0));

    float current_mutation_rate = getDynamicMutationRate(0, best_fitness, best_fitness);

    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        mt19937 &rnd = generators[tid];
        // split the iterations among threads with nowait to avoid synchronization
        #pragma omp for nowait
        for (int i = 0; i < POP_SIZE; ++i)
        {
            int p1 = selectParent(rnd);
            int p2 = selectParent(rnd);
            while (p1 == p2)
            {
                p2 = selectParent(rnd);
            }
            auto child = crossover(population[p1], population[p2], rnd);
            mutate(child, rnd, current_mutation_rate);
            new_population[i] = child;
        }
    }

    population = new_population;
}

// Print best route and distance
void printBestRoute()
{
    cout << "Best Route Distance: " << best_fitness << endl;
    cout << "Route: ";
    for (int i = 0; i < CITY_COUNT; ++i)
        cout << best_route[i] << " ";
    cout << best_route[0] << endl; // return to start
}

int main()
{
    
    // Set number of threads
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    fitness.resize(POP_SIZE);
    best_route.resize(CITY_COUNT);
    best_fitness = calculateTotalDistance(population[0]);
    best_route = population[0];

    cout<<"Initial Best Distance: "<<best_fitness<<endl;
    
    clock_t start = clock();

    for (int gen = 0; gen < GENERATIONS; ++gen)
    {
        evaluateFitness();//calculate total distance of each route
        nextGeneration();//generate new population

        if (gen % 1000 == 0)
        {
            #pragma omp master
            {
                cout << "Generation " << gen << " - Best Distance: " << best_fitness << endl;
            }
        }
    }

#pragma omp master
    {
        cout << "Generation " << GENERATIONS << " - Best Distance: " << best_fitness << endl;
    }

    clock_t end = clock();
    double exec_time = (double)(end - start) / CLOCKS_PER_SEC;

#pragma omp master
    {
        cout << "\nFinal Results:" << endl;
        printBestRoute();
        cout << "Execution Time: " << exec_time << " seconds\n";
    }

    double avg_fitness = 0.0;
#pragma omp parallel for reduction(+ : avg_fitness)
    for (int i = 0; i < POP_SIZE; ++i)
        avg_fitness += fitness[i];
    avg_fitness /= POP_SIZE;

#pragma omp master
    {
        cout << "Average Fitness (last generation): " << avg_fitness << endl;
    }

        
    
    return 0;
}
