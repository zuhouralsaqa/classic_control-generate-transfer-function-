% Genetic Algorithm Parameters
population_size = 10;
numGenerations = 500;
mutationRate = 0.08;
crossoverRate = 0.8;
numVariables = 6;
maxValue = 9;

% num of gens
individual_size = numVariables;


best_solutions = zeros(numGenerations, individual_size);
chromosome = zeros(1, numVariables);
population = randi(maxValue, population_size, individual_size);
best_fitnesses = zeros(numGenerations, 1);

for generation = 1:numGenerations
    
    % Evaluate the fitness of each individual
    fitness = evaluate_fitness(population);
    
    % best solution
    [~, bestIndex] = max(fitness);
    % best_solution(generation, 2) = fitness(bestIndex);
    best_solutions(generation, :) = population(bestIndex, :); 
    best_fitnesses(generation) = fitness(bestIndex);
    
    % display results
    fprintf('Generation %d: Best Solution = %s, Fitness = %f\n', generation, num2str(best_solutions(generation, :)), best_fitnesses(generation));
    
    % Select parents for crossover
    selected_parents = select_parents(population, fitness);
    
    crossed_over = cross_over(selected_parents, crossoverRate);
    
    mutated = mutation(crossed_over, mutationRate, maxValue);
    
    
    % define childrens, next generation
    population = mutated;
    
end


% one more time for the lat generation 

    % Evaluate the fitness of each individual
    fitness = evaluate_fitness(population);
    
    % best solution
    [~, bestIndex] = max(fitness);
    best_solutions(numGenerations, :) = population(bestIndex, :); 
    best_fitnesses(numGenerations) = fitness(bestIndex);
    
    % display results
    fprintf('Generation %d: Best Solution = %s, Fitness = %f\n', generation, num2str(best_solutions(numGenerations, :)),  fitness(bestIndex));



% sort all solutions in the last generation 

[sortedFitness, sortedIndices] = sort(fitness, 'descend');
sortedSolutions = population(sortedIndices, :);

% Display the last generation solutions
fprintf('\n Last generation:\n');
for i = 1:population_size
    fprintf('Solution %d: %s, Fitness = %f\n', i, num2str(sortedSolutions(i, :)), sortedFitness(i));
end


% display the best solution 
[~, bestGeneration] = max(best_fitnesses);
best_chromosome = best_solutions(bestGeneration, :);

chromosome = best_chromosome;
a = chromosome(1, 1);
b = chromosome(1, 2);
c = chromosome(1, 3);
d = chromosome(1, 4);
e = chromosome(1, 5);
f = chromosome(1, 6);

my_sys=tf([1 e f ],[1 a b c d]);
rlocus(my_sys);



% Define the selectParents function (you can implement various selection methods)
function selected_parents = select_parents(population, fitness)
    % Example: Roulette wheel selection
    totalFitness = sum(fitness);
    probabilities = fitness / totalFitness;
    
    % returs a vector of length = length(fitness) that takes the values
    % from 1 to last index in the population i.e., 10. according to the
    % probabilities given
    selectedIndices = randsample(1:length(fitness), length(fitness), true, probabilities);
    selected_parents = population(selectedIndices, :);
end



% Define the crossover function (you can implement various crossover methods)
function offspring = cross_over(parents, crossoverRate)
 
    % to have the same array size and clone the remaining chromosoms
    offspring = parents;
    numParents = size(parents, 1);
    numPairs = floor(numParents / 2);
    
    for i = 1:numPairs
        
        if rand() < crossoverRate
            
%             % Select two parents for crossover / complementary
%             parent1 = parents(i, :);
%             parent2 = parents((population_size - i + 1), :);
            
%             % make sure parent1 =! parent2
%             while ~(isequal(parent1,parent2))
%                 parent2 = parents(randi(numParents), :);
%             end

            parent1 = parents(2 * i - 1, :);
            parent2 = parents(2 * i, :);
            
            % Choose a random crossover point
            crossoverPoint = randi(length(parent1) - 1);
            
            % Perform single-point crossover
            offspring(2 * i - 1, :) = [parent1(1:crossoverPoint), parent2(crossoverPoint + 1:end)];
            offspring(2 * i, :) = [parent2(1:crossoverPoint), parent1(crossoverPoint + 1:end)];
            
        end
    end
end




% Define the mutate function
function mutatedOffspring = mutation(offspring, mutationRate, maxValue)
    
    % to have the same array size, and clone other chromosoms
    mutatedOffspring = offspring;
    [numOffspring, numGenes] = size(offspring);
    
    for i = 1:numOffspring
        for j = 1:numGenes
            if rand() < mutationRate
                mutatedOffspring(i, j) = randi(maxValue); % Flip the bit
            end
        end
    end
end



% Define the evaluateFitness function (customize this for your problem)
function fitness = evaluate_fitness(Population)

    % Example fitness function: sum of variables
    fitness_size = size(Population, 1);
    fitness = zeros(fitness_size, 1);
    
    for i = 1:fitness_size
        chromosome = Population(i, :);
        a = chromosome(1, 1);
        b = chromosome(1, 2);
        c = chromosome(1, 3);
        d = chromosome(1, 4);
        e = chromosome(1, 5);
        f = chromosome(1, 6);
        % kcr = chromosome(1, 7);
        
        k1 = (c * b - a * d - (c * c / a)) / (e - (e * e / a));
        k2 = (((c + e * b - 2 * c * (e / a) - a * f) / (2 * (e - ((e ^ 2) / a))))) ^ 2;
        % k2 is always positive, so is kcr.
        kcr = sqrt(k2); 
        s = k1 * 4 * ((e - (e^2 / a))^2);
        p = (c + e * b - 2 * c * (e/a) - a * f);
        
        eqn1 = (k1 == k2);
        eqn2 = (((c + e * b - 2 * c * (e / a) - a * f) / (2 * (e - (e ^ 2 / a))))) < 0;
        eqn3 = (a * b - c > ((a * a) * d) / c);
        eqn4 = e < a;
        eqn5 = a > 0;
        eqn6 = b > 0;
        eqn7 = c > 0;
        eqn8 = d > 0;
        eqn9 = e > 0;
        eqn10 = f > 0;
        eqn11 = k1 > 0;
        eqn12 = k2 > 0;
        
        % for k1 = kcr^2
        eqn13 = (abs(k1-(kcr^2)) == 0);
        % for sqrt(k1) = -sqrt(k2)
        eqn14 = (abs(sqrt(s)+p) == 0); 
        
        fitness_1 = eqn1 && eqn2 && eqn3 && eqn4 && eqn5 && eqn6 && eqn7 && eqn8 && eqn9 && eqn10 && eqn11 && eqn12 && eqn13 && eqn14;
        % to avoide inf values
        fitness_0_9 = fitness_1 - 0.1;

        fitness(i, 1) = -1 * log(1 - fitness_0_9) + 1;
        
    end
    
    
end