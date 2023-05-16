import random
import numpy as np

# задаем матрицу расстояний между городами
distances = np.array([
    [0, 1, 2, 3, 9],
    [1, 0, 4, 2, 6],
    [2, 4, 0, 5, 7],
    [3, 2, 5, 0, 8],
    [9, 6, 7, 8, 0]])


# функция оценки приспособленности
def fitness(genome):
    total_distance = 0
    for i in range(len(genome) - 1):
        current_city = genome[i]
        next_city = genome[i + 1]
        total_distance += distances[current_city][next_city]
    # добавляем расстояние от последнего города до города с номером 0
    total_distance += distances[genome[-1]][0]
    # добавляем расстояние от города с номером 0 до первого города в маршруте
    total_distance += distances[0][genome[0]]
    return 1 / total_distance


def tournament_selection(population, fitnesses, tournament_size=3):
    idx = random.sample(range(len(population)), tournament_size)
    tournament = [population[i] for i in idx]
    tournament_fitnesses = [fitnesses[i] for i in idx]
    winner_idx = tournament_fitnesses.index(max(tournament_fitnesses))
    return tournament[winner_idx]

def crossover(parent1, parent2):
    child = [-1] * len(parent1)
    # выбираем случайный участок генов от первого родителя
    start = random.randint(0, len(parent1)-1)
    end = random.randint(start, len(parent1)-1)
    child[start:end+1] = parent1[start:end+1]
    # заполняем оставшиеся гены из второго родителя
    for i in range(len(parent2)):
        if parent2[i] not in child:
            for j in range(len(child)):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break
    return child

def mutation(genome, mutation_rate=0.01):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(genome)-1)
            genome[i], genome[j] = genome[j], genome[i]
    return genome

def genetic_algorithm(num_cities, population_size, num_generations):
    # создаем начальную популяцию геномов
    population = []
    for i in range(population_size):
        genome = list(range(1, num_cities))
        random.shuffle(genome)
        population.append([0] + genome)
    for generatгion in range(num_generations):
        # оцениваем каждый геном в популяции
        fitnesses = [fitness(genome) for genome in population]

        # выбираем лучших геномов для скрещивания
        selected = [tournament_selection(population, fitnesses) for i in range(population_size)]
        # скрещиваем и мутируем выбранных геномов
        offspring = []
        for i in range(0, population_size, 2):
            child1 = crossover(selected[i], selected[i + 1])
            child2 = crossover(selected[i + 1], selected[i])
            child1 = mutation(child1)
            child2 = mutation(child2)
            offspring.append(child1)
            offspring.append(child2)

        # заменяем старых геномов в популяции на новых
        population = offspring

    # оцениваем каждый геном в конечной популяции
    fitnesses = [fitness(genome) for genome in population]
    best_idx = np.argmax(fitnesses)
    best_genome = population[best_idx]
    best_fitness = fitnesses[best_idx]

    # выводим результаты
    print("Лучший геном: ", best_genome)
    print("Приспособленность: ", best_fitness)
    # вычисляем стоимость маршрута
    total_distance = 0
    for i in range(len(best_genome) - 1):
        current_city = best_genome[i]
        next_city = best_genome[i + 1]
        total_distance += distances[current_city][next_city]
    total_distance += distances[best_genome[-1]][0]
    print("Стоимость маршрута: ", total_distance)
    return best_genome


best_route = genetic_algorithm(num_cities=5, population_size=1000, num_generations=1000)
print(best_route)

