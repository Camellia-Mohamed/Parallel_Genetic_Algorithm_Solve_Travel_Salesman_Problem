#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <omp.h>
#include <random>

using namespace std;

#include "data.h"

vector<double> fitness; // the total distance of the route
vector<int> best_route;
double best_fitness = 1e9;

// Thread-safe way to update best route
void updateBestRoute(const vector<int> &route, double route_fitness)
{
#pragma omp critical
    {
        if (route_fitness < best_fitness)
        {
            best_fitness = route_fitness;
            best_route = route;
        }
    }
}

double distance(const City &a, const City &b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// calculate the total distance of the route
double calculateTotalDistance(const vector<int> &route)
{
    if (route.size() != CITY_COUNT)
        return 1e9;

    double dist = 0;
#pragma omp parallel for reduction(+ : dist)
    for (int i = 0; i < CITY_COUNT - 1; ++i)
    {
        if (route[i] >= 0 && route[i] < CITY_COUNT &&
            route[i + 1] >= 0 && route[i + 1] < CITY_COUNT)
        {
            dist += distance(cities[route[i]], cities[route[i + 1]]);
        }
        else
        {
            dist = 1e9;
        }
    }

    // Add the return distance to start city
    if (route[CITY_COUNT - 1] >= 0 && route[CITY_COUNT - 1] < CITY_COUNT &&
        route[0] >= 0 && route[0] < CITY_COUNT)
    {
        dist += distance(cities[route[CITY_COUNT - 1]], cities[route[0]]);
    }
    else
    {
        return 1e9;
    }
    return dist;
}

int selectParent(mt19937 &gen)
{
    uniform_int_distribution<> dist(0, POP_SIZE - 1);
    int best = dist(gen);
    for (int i = 1; i < SELECTION_SIZE; ++i)
    {
        int idx = dist(gen);
        if (fitness[idx] < fitness[best])
            best = idx;
    }
    return best;
}

// Dynamic mutation rate that increases when stuck
float getDynamicMutationRate(int generation, double current_best, double previous_best)
{
    static double last_improvement = 0;
    static int generations_without_improvement = 0;

    if (current_best < previous_best)
    {
        last_improvement = current_best;
        generations_without_improvement = 0;
        return MUTATION_RATE;
    }

    generations_without_improvement++;
    return min(0.5, MUTATION_RATE * (1.0 + generations_without_improvement / 1000.0));
}

// Order Crossover (OX) with random crossover points
vector<int> crossover(const vector<int> &parent1, const vector<int> &parent2, mt19937 &gen)
{
    if (parent1.size() != CITY_COUNT || parent2.size() != CITY_COUNT)
        return parent1;

    vector<int> child(CITY_COUNT, -1);
    vector<bool> used(CITY_COUNT, false);

    // Use random crossover points instead of fixed ones
    uniform_int_distribution<> dist(0, CITY_COUNT - 1);
    int start = dist(gen);
    int end = dist(gen);
    if (start > end)
        swap(start, end);

    // Copy the segment from parent1
    for (int i = start; i <= end && i < CITY_COUNT; ++i)
    {
        if (parent1[i] >= 0 && parent1[i] < CITY_COUNT)
        {
            child[i] = parent1[i];
            used[parent1[i]] = true;
        }
    }

    // Fill the rest from parent2
    int j = 0;
    for (int i = 0; i < CITY_COUNT; ++i)
    {
        if (i >= start && i <= end)
            continue;

        while (j < CITY_COUNT && (used[parent2[j]] || parent2[j] < 0 || parent2[j] >= CITY_COUNT))
            ++j;

        if (j < CITY_COUNT)
        {
            child[i] = parent2[j];
            used[parent2[j]] = true;
            ++j;
        }
    }

    // Verify the child is valid
    for (int i = 0; i < CITY_COUNT; ++i)
    {
        if (child[i] < 0 || child[i] >= CITY_COUNT)
            return parent1;
    }

    return child;
}

// Enhanced mutation with multiple swap operations
void mutate(vector<int> &route, mt19937 &gen, float mutation_rate)
{
    uniform_real_distribution<> prob(0.0, 1.0);
    if (prob(gen) < mutation_rate)
    {
        uniform_int_distribution<> dist(0, CITY_COUNT - 1);
        // Perform multiple swaps based on mutation rate
        int num_swaps = max(1, int(mutation_rate * 5));
        for (int s = 0; s < num_swaps; ++s)
        {
            int i = dist(gen);
            int j = dist(gen);
            swap(route[i], route[j]);
        }
    }
}

void evaluateFitness()
{
    if (fitness.size() != POP_SIZE)
        fitness.resize(POP_SIZE);

    double local_best_fitness = 1e9;
    vector<int> local_best_route;
    local_best_route.resize(CITY_COUNT);

#pragma omp parallel
    {
        double thread_best_fitness = 1e9;
        vector<int> thread_best_route;
        thread_best_route.resize(CITY_COUNT);

#pragma omp for schedule(dynamic, 10) nowait
        for (int i = 0; i < POP_SIZE; ++i)
        {
            if (i < population.size() && population[i].size() == CITY_COUNT)
            {
                double fit = calculateTotalDistance(population[i]);
                fitness[i] = fit;

                if (fit < thread_best_fitness)
                {
                    thread_best_fitness = fit;
                    thread_best_route = population[i];
                }
            }
        }

#pragma omp critical
        {
            if (thread_best_fitness < local_best_fitness)
            {
                local_best_fitness = thread_best_fitness;
                local_best_route = thread_best_route;
            }
        }
    }

    updateBestRoute(local_best_route, local_best_fitness);
}

void nextGeneration()
{
    if (population.size() != POP_SIZE)
        return;

    vector<vector<int>> new_population(POP_SIZE);
    for (auto &route : new_population)
    {
        route.resize(CITY_COUNT);
    }

    // Get dynamic mutation rate
    float current_mutation_rate = getDynamicMutationRate(0, best_fitness, best_fitness);

#pragma omp parallel
    {
        // Each thread gets its own random number generator
        random_device rd;
        mt19937 gen(rd());

#pragma omp for schedule(dynamic, 10)
        for (int i = 0; i < POP_SIZE; ++i)
        {
            try
            {
                int p1 = selectParent(gen);
                int p2 = selectParent(gen);

                if (p1 < population.size() && p2 < population.size())
                {
                    auto child = crossover(population[p1], population[p2], gen);
                    mutate(child, gen, current_mutation_rate);
                    new_population[i] = std::move(child);
                }
                else
                {
                    new_population[i] = population[i % population.size()];
                }
            }
            catch (const exception &e)
            {
                cerr << "Error in thread " << omp_get_thread_num()
                     << " at index " << i << ": " << e.what() << endl;
                new_population[i] = population[i % population.size()];
            }
        }
    }

    population = std::move(new_population);
}

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
    try
    {
        int num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);

        fitness.resize(POP_SIZE);
        best_route.resize(CITY_COUNT);

        // Initialize best_fitness with first route
        best_fitness = calculateTotalDistance(population[0]);
        best_route = population[0];

        cout << "Starting Parallel Genetic Algorithm..." << endl;
        cout << "Number of Threads: " << num_threads << endl;
        cout << "Population Size: " << POP_SIZE << endl;
        cout << "Number of Cities: " << CITY_COUNT << endl;
        cout << "Number of Generations: " << GENERATIONS << endl;
        cout << "Initial Best Distance: " << best_fitness << endl;

        double start = omp_get_wtime();

        for (int gen = 0; gen < GENERATIONS; ++gen)
        {
            try
            {
                evaluateFitness();
                nextGeneration();

                if (gen % 50 == 0)
                {
                    cout << "Generation " << gen << " - Best Distance: " << best_fitness << endl;
                }
            }
            catch (const exception &e)
            {
                cerr << "Error in generation " << gen << ": " << e.what() << endl;
            }
        }
        cout << "Generation " << GENERATIONS << " - Best Distance: " << best_fitness << endl;

        double end = omp_get_wtime();
        double execution_time = end - start;

        cout << "\nFinal Results:" << endl;
        printBestRoute();
        cout << "Execution Time: " << execution_time << " seconds\n";

        double avg_fitness = 0.0;
#pragma omp parallel for reduction(+ : avg_fitness) schedule(dynamic, 10)
        for (int i = 0; i < POP_SIZE; ++i)
            avg_fitness += fitness[i];
        avg_fitness /= POP_SIZE;

        cout << "Average Fitness (last generation): " << avg_fitness << endl;
    }
    catch (const exception &e)
    {
        cerr << "Error occurred: " << e.what() << endl;
        return 1;
    }
    catch (...)
    {
        cerr << "Unknown error occurred" << endl;
        return 1;
    }

    return 0;
}