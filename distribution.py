import math
import random

def unit_sphere_distribution_sample(size):
    

    sphere_distribution = [random.random() for i in range(size)]

    mag = math.sqrt(sum(i ** 2 for i in sphere_distribution))

    sphere = [i / mag for i in sphere_distribution]

    return sphere


def boolean_distribution_sample(size):
    
    return [random.randint(0, 1) for i in range(size)]


ALLOWED_DISTRIBUTIONS = {
    "bool": boolean_distribution_sample,
    "sphere": unit_sphere_distribution_sample
}