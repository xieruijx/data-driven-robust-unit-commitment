import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], size=len(bounds))
        self.velocity = np.random.uniform(-1, 1, size=len(bounds))
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        r1, r2 = np.random.rand(2)
        self.velocity = (inertia_weight * self.velocity +
                         cognitive_weight * r1 * (self.best_position - self.position) +
                         social_weight * r2 * (global_best_position - self.position))

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

def pso(loss_function, num_particles=10, bounds=(0, 1), num_iterations=4, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.5):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')

    for iteration in range(num_iterations):
        print(f"Begin Iteration: {iteration}")
        for particle in particles:
            value = loss_function(particle.position)

            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = np.copy(particle.position)

            if value < global_best_value:
                global_best_value = value
                global_best_position = np.copy(particle.position)

            print("Best Position:", global_best_position)
            print("Best Value:", global_best_value)

        for particle in particles:
            particle.update_velocity(global_best_position, inertia_weight, cognitive_weight, social_weight)
            particle.update_position(bounds)

    return global_best_position, global_best_value
