import numpy as np
# n-dimensional points
class Point:
    def __init__(self,coordinates):
        '''
        :coordinates: a list of coordinates (n dimensions)
        '''
        self.coordinates = coordinates
    def diff(self,other):
        assert len(self.coordinates) == len(other.coordinates), "Not equal length."
        return [self.coordinates[i]-other.coordinates[i]\
                for i in range(len(self.coordinates))]
    def dist(self,other):
        assert len(self.coordinates) == len(other.coordinates), "Not equal length."
        total_sum = 0
        for i in range(len(self.coordinates)):
            total_sum += (self.coordinates[i] - other.coordinates[i])**2
        return np.sqrt(total_sum)

class Polygon:
    def __init__(self,points):
        self.points = points
    def herons_formula(points):
        assert len(points) == 3, "Not a valid triangle."
        side1 = points[0].dist(points[1])
        side2 = points[1].dist(points[2])
        side3 = points[2].dist(points[0])
        semi_perimeter = (side1+side2+side3)/2
        return np.sqrt(semi_perimeter*\
                      (semi_perimeter-side1)*\
                      (semi_perimeter-side2)*\
                      (semi_perimeter-side3))
    def compute_area(self):
        assert len(self.points) >= 3, "Not enough points for convex polygon."
        # having the same dir
        line_dirs = dict()
        # direction vectors
        change_ratio = None
        total_area = 0
        for i in range(1,len(self.points)):
            if i < len(self.points)-1:
                # assuming points are valid.
                total_area += Polygon.herons_formula(self.points[i-1:i+2])
        return total_area

# On a 10x10x10 Cube - None of the points are on
# the bottom face or along the edges.
#      #
#      #
#      #
#      # # # # #
#    #
#  #

#             ####### 0,0
#             #   x #
#     ####### ####### #######
#     #     # #    x# #  x  #
#     ####### ####### #######
#             #     #
#             #######
# If next face is adjacent, then compute the correct distance of the flattened
# solid.
#

def within_cube(point, dim):
    return all([0 <= val and val <= dim for val in point])


def dist_cube(coordinate1, coordinate2):
    assert len(coordinate1) == len(coordinate2), "Not equal length."
    total_sum = 0
    for i in range(len(coordinate1)):
        total_sum += (coordinate1[i] - coordinate2[i])**2
    return round(np.sqrt(total_sum),2)

def on_same_face(coordinate1, coordinate2):
    common_entries = 0
    for i in range(len(coordinate1)):
        common_entries += (coordinate1[i] == coordinate2[i])
    return common_entries == 2



def flattened_distance(coordinate1, coordinate2):
    to_or_from_top = (coordinate1[2] == 10) or (coordinate2[2] == 10)
    if to_or_from_top:
        if (coordinate1[1] in {0,10} or coordinate2[1] in {0,10}):
            val = round(np.sqrt(((abs(coordinate1[0]-coordinate2[0]))**2)+\
                  ((abs(coordinate1[1]-coordinate2[1])+\
                    abs(coordinate1[2]-coordinate2[2]))**2)),2)
        else:
            # assumed on x=0 or x=10 face.
            assert (coordinate1[0] in {0,10} or coordinate2[0] in {0,10}),\
                "Not on face"
            val = round(np.sqrt(((coordinate1[1]-coordinate2[1])**2)+\
                  ((abs(coordinate1[2]-coordinate2[2])+\
                    abs(coordinate1[0]-coordinate2[0]))**2)),2)
    else:
        val = round(np.sqrt(((abs(coordinate1[0]-coordinate2[0])+\
                              abs(coordinate1[1]-coordinate2[1]))**2)+\
                ((coordinate1[2]-coordinate2[2])**2)), 2)
    return val
# assumed that the points are in 3D and on the same face
# assuming can go under surface
def on_a_cube(filepath,only_surface=True):
    coordinates = []
    distance = 0
    with open(filepath,"r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                N = int(line)
            else:
                assert i == 1, "Not just two lines."
                points = [int(val) for val in line.split(",")]
                for i in range(0,len(points),3):
                    coordinates.append((points[i],points[i+1],points[i+2]))
    for j in range(len(coordinates)-1):
        check = on_same_face(coordinates[j], coordinates[j+1])
        # 60 degrees
        distance += round(np.pi/3*dist_cube(coordinates[j],coordinates[j+1])\
                     if check else (flattened_distance(coordinates[j],
                                                       coordinates[j+1])\
                                    if only_surface else\
                                    dist_cube(coordinates[j],
                                              coordinates[j+1])),2)
    return distance

if __name__ == "__main__":
    print(on_a_cube("./input2.txt"))
    p = Polygon([Point([1,2]),Point([3,4]),Point([8,-2]),Point([5,-5])])
    print(p.compute_area())
