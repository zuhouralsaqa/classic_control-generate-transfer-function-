
function guiProgram()
 
    % Create a figure and set its size
    % Create a figure window
    fig = figure('Position', [100 100 800 400]);
 
%     % Create UI components
%     aLabel = uicontrol('Style', 'text', 'String', 'a:', 'Position', [20 350 30 20]);
%     aEdit = uicontrol('Style', 'edit', 'Position', [60 350 80 20]);
%     bLabel = uicontrol('Style', 'text', 'String', 'b:', 'Position', [20 320 30 20]);
%     bEdit = uicontrol('Style', 'edit', 'Position', [60 320 80 20]);
%     cLabel = uicontrol('Style', 'text', 'String', 'c:', 'Position', [20 290 30 20]);
%     cEdit = uicontrol('Style', 'edit', 'Position', [60 290 80 20]);
%     dLabel = uicontrol('Style', 'text', 'String', 'd:', 'Position', [20 260 30 20]);
%     dEdit = uicontrol('Style', 'edit', 'Position', [60 260 80 20]);
%     eLabel = uicontrol('Style', 'text', 'String', 'e:', 'Position', [20 230 30 20]);
%     eEdit = uicontrol('Style', 'edit', 'Position', [60 230 80 20]);
%     fLabel = uicontrol('Style', 'text', 'String', 'f:', 'Position', [20 200 30 20]);
%     fEdit = uicontrol('Style', 'edit', 'Position', [60 200 80 20]);
%     kcrLabel = uicontrol('Style', 'text', 'String', 'kcr:', 'Position', [20 170 30 20]);
%     kcrEdit = uicontrol('Style', 'edit', 'Position', [60 170 80 20]);
    solveButton = uicontrol('Style', 'pushbutton', 'String', 'Generate', 'Position', [150 180 80 30]);
    rootLocusAxes = axes('Parent', fig, 'Position', [0.55 0.15 0.4 0.6]);
    tf_label = uicontrol('Style', 'text', 'Position', [470 310 250 80], 'String', '', 'FontSize', 12);
    %display_button = uicontrol('Style', 'pushbutton', 'Position', [150 10 100 30], 'String', 'Display TF', 'Callback', @displayTransferFunction);
 
 
    % Callback function for the Solve button
    solveButton.Callback = @solveButtonCallback;
 
    % Callback function for the Solve button
    function solveButtonCallback(~, ~)
        
        % write your func here
        
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
        
        fprintf('\n best solution:\n');
        fprintf('Generation %d: Best Solution = %s, Fitness = %f\n', bestGeneration, num2str(best_solutions(bestGeneration, :)), best_fitnesses(bestGeneration));

        best_a = best_chromosome(1, 1);
        best_b = best_chromosome(1, 2);
        best_c = best_chromosome(1, 3);
        best_d = best_chromosome(1, 4);
        best_e = best_chromosome(1, 5);
        best_f = best_chromosome(1, 6);

        my_sys=tf([1 best_e best_f ],[1 best_a best_b best_c best_d]);
        rlocus(my_sys);


        % Define the transfer function coefficients
        num = [1 best_e best_f];
        den = [1 best_a best_b best_c best_d];
 
        % Create the transfer function object
        transfer_func = tf(num, den);
 
        % Display the transfer function in the GUI
        set(tf_label, 'String', transferFunctionToString(transfer_func));
    end
 
    % Function to convert the transfer function to a string
    function tf_str = transferFunctionToString(transfer_func)
        % Get the numerator and denominator coefficients
        num_coeffs = transfer_func.Numerator{1};
        den_coeffs = transfer_func.Denominator{1};
 
        % Convert the coefficients to strings
        num_str = poly2str(num_coeffs, 's');
        den_str = poly2str(den_coeffs, 's');
 
        % Construct the transfer function string
        tf_str = sprintf('\n            %s\n------------------------------------\n%s', num_str, den_str);
        
    end






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

            for j = 1:numPairs

                if rand() < crossoverRate

                    parent1 = parents(2 * j - 1, :);
                    parent2 = parents(2 * j, :);

                    % Choose a random crossover point
                    crossoverPoint = randi(length(parent1) - 1);

                    % Perform single-point crossover
                    offspring(2 * j - 1, :) = [parent1(1:crossoverPoint), parent2(crossoverPoint + 1:end)];
                    offspring(2 * j, :) = [parent2(1:crossoverPoint), parent1(crossoverPoint + 1:end)];

                end
            end
        end




        % Define the mutate function
        function mutatedOffspring = mutation(offspring, mutationRate, maxValue)

            % to have the same array size, and clone other chromosoms
            mutatedOffspring = offspring;
            [numOffspring, numGenes] = size(offspring);

            for q = 1:numOffspring
                for j = 1:numGenes
                    if rand() < mutationRate
                        mutatedOffspring(q, j) = randi(maxValue); % Flip the bit
                    end
                end
            end
        end



        % Define the evaluateFitness function (customize this for your problem)
        function fitness = evaluate_fitness(Population)

            % Example fitness function: sum of variables
            fitness_size = size(Population, 1);
            fitness = zeros(fitness_size, 1);

            for w = 1:fitness_size
                chromosome = Population(w, :);
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

                fitness(w, 1) = -1 * log(1 - fitness_0_9) + 1;

            end


        end
   
end
