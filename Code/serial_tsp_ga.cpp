#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>

using namespace std;

#include "data.h"

vector<double> fitness; // the total distance of the route
vector<int> best_route;
double best_fitness = 1e9;

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
    for (int i = 0; i < CITY_COUNT - 1; ++i)
    {
        if (route[i] >= 0 && route[i] < CITY_COUNT &&
            route[i + 1] >= 0 && route[i + 1] < CITY_COUNT)
        {
            dist += distance(cities[route[i]], cities[route[i + 1]]);
        }
        else
        {
            return 1e9;
        }
    }
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

int selectParent()
{
    if (POP_SIZE == 0)
        return 0;

    int best = 0;
    int selection_count = min(SELECTION_SIZE, POP_SIZE);

    for (int i = 0; i < selection_count; ++i)
    {
        if (fitness[i] < fitness[best])
            best = i;
    }
    return best;
}

vector<int> crossover(const vector<int> &parent1, const vector<int> &parent2)
{
    if (parent1.size() != CITY_COUNT || parent2.size() != CITY_COUNT)
        return parent1;

    vector<int> child(CITY_COUNT, -1);
    vector<bool> used(CITY_COUNT, false);

    // Copy the segment from parent1
    for (int i = CROSSOVER_START; i <= CROSSOVER_END && i < CITY_COUNT; ++i)
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
        if (i >= CROSSOVER_START && i <= CROSSOVER_END)
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
            return parent1; // Return parent1 if child is invalid
    }

    return child;
}

void mutate(vector<int> &route)
{
    if (route.size() != CITY_COUNT)
        return;

    if (MUTATION_POINT1 >= 0 && MUTATION_POINT1 < CITY_COUNT &&
        MUTATION_POINT2 >= 0 && MUTATION_POINT2 < CITY_COUNT)
    {
        swap(route[MUTATION_POINT1], route[MUTATION_POINT2]);
    }
}

void evaluateFitness()
{
    if (fitness.size() != POP_SIZE)
        fitness.resize(POP_SIZE);

    for (int i = 0; i < POP_SIZE; ++i)
    {
        if (i < population.size() && population[i].size() == CITY_COUNT)
        {
            double fit = calculateTotalDistance(population[i]);
            fitness[i] = fit;

            if (fit < best_fitness)
            {
                best_fitness = fit;
                best_route = population[i];
            }
        }
    }
}

void nextGeneration()
{
    if (population.size() != POP_SIZE)
        return;

    vector<vector<int>> new_population(POP_SIZE);

    for (int i = 0; i < POP_SIZE; ++i)
    {
        int p1 = selectParent();
        int p2 = selectParent();

        if (p1 < population.size() && p2 < population.size())
        {
            auto child = crossover(population[p1], population[p2]);
            mutate(child);
            new_population[i] = child;
        }
        else
        {
            new_population[i] = population[i % population.size()];
        }
    }

    population = new_population;
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
    // cout<<population.size();
    try
    {
        // Initialize fitness vector with proper size
        fitness.resize(POP_SIZE);
        best_route.resize(CITY_COUNT);

        // Initialize best_fitness with first route
        best_fitness = calculateTotalDistance(population[0]);
        best_route = population[0];

        cout << "Starting Genetic Algorithm..." << endl;
        cout << "Population Size: " << POP_SIZE << endl;
        cout << "Number of Cities: " << CITY_COUNT << endl;
        cout << "Number of Generations: " << GENERATIONS << endl;
        cout << "Initial Best Distance: " << best_fitness << endl;

        clock_t start = clock();

        for (int gen = 0; gen < GENERATIONS; ++gen)
        {
            evaluateFitness();
            nextGeneration();

            // Print progress every 50 generations
            if (gen % 50 == 0)
            {
                cout << "Generation " << gen << " - Best Distance: " << best_fitness << endl;
            }
        }

        clock_t end = clock();
        double execution_time = (double)(end - start) / CLOCKS_PER_SEC;

        cout << "\nFinal Results:" << endl;
        printBestRoute();
        cout << "Execution Time: " << execution_time << " seconds\n";

        // Compute average fitness
        double avg_fitness = 0.0;
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
