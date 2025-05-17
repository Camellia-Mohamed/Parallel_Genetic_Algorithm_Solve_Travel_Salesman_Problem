#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <random>

using namespace std;

#include "data.h" // Make sure this defines: cities vector, CITY_COUNT, POP_SIZE, GENERATIONS, MUTATION_RATE, SELECTION_SIZE

vector<double> fitness; // total distance of each route
vector<int> best_route;
double best_fitness = 1e9;

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

double distance(const City &a, const City &b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// Calculate total route distance
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
    // Return to start city
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

// Randomized tournament selection
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

// Evaluate fitness of entire population and update best route
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

// Generate next generation
void nextGeneration()
{
    if (population.size() != POP_SIZE)
        return;

    vector<vector<int>> new_population(POP_SIZE);

    random_device rd;
    mt19937 gen(rd());

    // Get dynamic mutation rate
    float current_mutation_rate = getDynamicMutationRate(0, best_fitness, best_fitness);

    for (int i = 0; i < POP_SIZE; ++i)
    {
        int p1 = selectParent(gen);
        int p2 = selectParent(gen);

        auto child = crossover(population[p1], population[p2], gen);
        mutate(child, gen, current_mutation_rate);
        new_population[i] = child;
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
    try
    {
        fitness.resize(POP_SIZE);
        best_route.resize(CITY_COUNT);

        // Initialize best_fitness and route with first population member
        best_fitness = calculateTotalDistance(population[0]);
        best_route = population[0];

        cout << "Initial Best Distance: " << best_fitness << endl;

        clock_t start = clock();

        for (int gen = 0; gen < GENERATIONS; ++gen)
        {
            evaluateFitness();
            nextGeneration();

            if (gen % 1000 == 0)
                cout << "Generation " << gen << " - Best Distance: " << best_fitness << endl;
        }

        cout << "Generation " << GENERATIONS << " - Best Distance: " << best_fitness << endl;

        clock_t end = clock();
        double exec_time = (double)(end - start) / CLOCKS_PER_SEC;

        cout << "\nFinal Results:" << endl;
        printBestRoute();
        cout << "Execution Time: " << exec_time << " seconds\n";

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
