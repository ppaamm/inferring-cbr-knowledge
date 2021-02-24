import retrieval
import analogy
import sys

class CBRAgent:
    
    
    def __init__(self, distance, analogy, k_neighbors):
        self.CB = []
        self.distance = distance
        self.k_neighbors = k_neighbors
        self.analogy = analogy
        
    
    def insert_case(self, new_problem, new_solution):
        self.CB.append([new_problem, new_solution])
    
    
    
    
    
    ###########################################################################
    ####   4 Rs
    ###########################################################################
    
    
    def retrieval(self, new_problem):
        return retrieval.retrieval_k(self.CB, new_problem, self.distance, self.k_neighbors)
    
    def reuse(self, new_problem, neighbor_cases):
        solutions = {}
        for source_case in neighbor_cases:
            results, length = analogy.solveAnalogy(source_case[0][0], source_case[0][1], new_problem)
            results = {r[0] for r in results}
            for r in results:
                if r in solutions:
                    solutions[r][0] += 1
                    solutions[r][1] = min(solutions[r][1], length)
                else:
                    solutions[r] = [1, length]
                    
        # Decision: 
        nb_occurences = 0
        majoritary_solutions = []
        for s in solutions:
            if solutions[s][0] > nb_occurences:
                nb_occurences = solutions[s][0]
                majoritary_solutions = [s]
            elif solutions[s][0] == nb_occurences:
                majoritary_solutions.append(s)
        
        if len(majoritary_solutions) == 1: return majoritary_solutions[0]
        
        # if multiple solutions
        # select the one with minimal length
        print("Conflict: solving using complexity")
        min_length = sys.maxsize
        solution = ""
        #TODO: What if two solutions have same complexity
        for s in majoritary_solutions:
            if solutions[s][1] < min_length:
                solution = s
                min_length = solutions[s][1]
        return solution
    
    
    #TODO: Have a retain policy
    def retain(self, new_problem, new_solution):
        self.insert_case(new_problem, new_solution)
        return True
    
    
    
    
    def solve_problem(self, new_problem):
        neighbor_cases = self.retrieval(new_problem)
        return self.reuse(new_problem, neighbor_cases)
        


################## Example
        
cbr = CBRAgent(retrieval.dist5, analogy.solveAnalogy, 2)
cbr.insert_case("rosa", "rosam")
cbr.insert_case("dominus", "dominum")
cbr.insert_case("corpus", "corpus")

print(cbr.solve_problem("vita"))
print(cbr.solve_problem("lingus"))
            
            
        