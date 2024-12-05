import numpy as np

beta= 10**20

def MKP1(x):
    # D = 28  # et 2 contraintes
    yyy = np.array([1898, 440, 22507, 270, 14148, 3100, 4650, 30800, 615, 4975, 1160, 4225, 510, 11880, 479, 440, 490, 330, 110, 560, 24355, 2885, 11748, 4550, 750, 3720, 1950, 10500])
    ccc = np.array([600, 600])
    A = np.array([[45, 0, 85, 150, 65, 95, 30, 0, 170, 0, 40, 25, 20, 0, 0, 25, 0, 0, 25, 0, 165, 0, 85, 0, 0, 0, 0, 100],
                  [30, 20, 125, 5, 80, 25, 35, 73, 12, 15, 15, 40, 5, 10, 10, 12, 10, 9, 0, 20, 60, 40, 50, 36, 49, 40, 19, 150]])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z

def MKP2(x):
    # D = 28  # et 2 contraintes
    yyy = np.array([1898, 440, 22507, 270, 14148, 3100, 4650, 30800, 615, 4975, 1160, 4225, 510, 11880, 479, 440, 490, 330, 110, 560, 24355, 2885, 11748, 4550, 750, 3720, 1950, 10500])
    ccc = np.array([500, 500])
    A = np.array([[45, 0, 85, 150, 65, 95, 30, 0, 170, 0, 40, 25, 20, 0, 0, 25, 0, 0, 25, 0, 165, 0, 85, 0, 0, 0, 0, 100],
              [30, 20, 125, 5, 80, 25, 35, 73, 12, 15, 15, 40, 5, 10, 10, 12, 10, 9, 0, 20, 60, 40, 50, 36, 49, 40, 19, 150]])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z

def MKP3(x):
    # D = 28  # et 2 contraintes
    yyy = np.array([1898, 440, 22507, 270, 14148, 3100, 4650, 30800, 615, 4975, 1160, 4225, 510, 11880, 479, 440, 490, 330, 110, 560, 24355, 2885, 11748, 4550, 750, 3720, 1950, 10500])
    ccc = np.array([300, 300])
    A = np.array([[45, 0, 85, 150, 65, 95, 30, 0, 170, 0, 40, 25, 20, 0, 0, 25, 0, 0, 25, 0, 165, 0, 85, 0, 0, 0, 0, 100],
                  [30, 20, 125, 5, 80, 25, 35, 73, 12, 15, 15, 40, 5, 10, 10, 12, 10, 9, 0, 20, 60, 40, 50, 36, 49, 40, 19, 150]])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z

def MKP4(x):
    # D = 28  # et 2 contraintes
    yyy = np.array([1898, 440, 22507, 270, 14148, 3100, 4650, 30800, 615, 4975, 1160, 4225, 510, 11880, 479, 440, 490, 330, 110, 560, 24355, 2885, 11748, 4550, 750, 3720, 1950, 10500])
    ccc = np.array([300, 600])
    A = np.array([[45, 0, 85, 150, 65, 95, 30, 0, 170, 0, 40, 25, 20, 0, 0, 25, 0, 0, 25, 0, 165, 0, 85, 0, 0, 0, 0, 100],
                  [30, 20, 125, 5, 80, 25, 35, 73, 12, 15, 15, 40, 5, 10, 10, 12, 10, 9, 0, 20, 60, 40, 50, 36, 49, 40, 19, 150]])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z

def MKP5(x):
    # D = 28  # et 2 contraintes
    yyy = np.array([1898, 440, 22507, 270, 14148, 3100, 4650, 30800, 615, 4975, 1160, 4225, 510, 11880, 479, 440, 490, 330, 110, 560, 24355, 2885, 11748, 4550, 750, 3720, 1950, 10500])
    ccc = np.array([600, 300])
    A = np.array([[45, 0, 85, 150, 65, 95, 30, 0, 170, 0, 40, 25, 20, 0, 0, 25, 0, 0, 25, 0, 165, 0, 85, 0, 0, 0, 0, 100],
                  [30, 20, 125, 5, 80, 25, 35, 73, 12, 15, 15, 40, 5, 10, 10, 12, 10, 9, 0, 20, 60, 40, 50, 36, 49, 40, 19, 150]])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z

def MKP6(x):
    # D = 28  # et 2 contraintes
    yyy = np.array([1898, 440, 22507, 270, 14148, 3100, 4650, 30800, 615, 4975, 1160, 4225, 510, 11880, 479, 440, 490, 330, 110, 560, 24355, 2885, 11748, 4550, 750, 3720, 1950, 10500])
    ccc = np.array([562, 497])
    A = np.array([[45, 0, 85, 150, 65, 95, 30, 0, 170, 0, 40, 25, 20, 0, 0, 25, 0, 0, 25, 0, 165, 0, 85, 0, 0, 0, 0, 100],
                  [30, 20, 125, 5, 80, 25, 35, 73, 12, 15, 15, 40, 5, 10, 10, 12, 10, 9, 0, 20, 60, 40, 50, 36, 49, 40, 19, 150]])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z

def MKP7(x):
    # D = 105  # et 2 contraintes
    yyy = np.array([41850, 38261, 23800, 21697, 7074, 5587, 5560, 5500, 3450, 2391, 761, 460, 367, 24785, 47910, 30250, 107200, 4235, 9835, 9262, 15000, 6399, 6155, 10874, 37100, 27040, 4117, 32240, 1600, 4500, 70610, 6570, 15290, 23840, 16500, 7010, 16020, 8000, 31026, 2568, 2365, 4350, 1972, 4975, 29400, 7471, 2700, 3840, 22400, 3575, 13500, 1125, 11950, 12753, 10568, 15600, 20652, 13150, 2900, 1790, 4970, 5770, 8180, 2450, 7140, 12470, 6010, 16000, 11100, 11093, 4685, 2590, 11500, 5820, 2842, 5000, 3300, 2800, 5420, 900, 13300, 8450, 5300, 750, 1435, 2100, 7215, 2605, 2422, 5500, 8550, 2700, 540, 2550, 2450, 725, 445, 700, 1720, 2675, 220, 300, 405, 150, 70])
    ccc = np.array([3000, 3000])
    A = np.array([
        [75, 40, 365, 95, 25, 17, 125, 20, 22, 84, 75, 50, 15, 0, 0, 12, 0, 10, 0, 50, 0, 0, 10, 0, 0, 50, 60, 150, 0, 0, 75, 0, 102, 0, 0, 40, 60, 0, 165, 0, 0, 0, 45, 0, 0, 0, 25, 0, 150, 0, 0, 0, 158, 0, 85, 95, 0, 89, 20, 0, 0, 0, 0, 0, 0, 80, 0, 110, 0, 15, 0, 60, 5, 135, 0, 0, 25, 0, 300, 35, 100, 0, 0, 25, 0, 0, 225, 25, 0, 0, 0, 0, 0, 0, 0, 5, 0, 60, 0, 100, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 50, 2, 5, 5, 10, 5, 6, 11, 41, 30, 5, 40, 2, 6, 100, 10, 25, 39, 30, 13, 30, 15, 60, 5, 5, 10, 5, 15, 91, 24, 10, 15, 90, 15, 60, 5, 55, 60, 50, 75, 100, 65, 15, 10, 30, 35, 50, 15, 45, 80, 40, 110, 80, 80, 36, 20, 90, 50, 25, 50, 35, 30, 60, 10, 150, 110, 70, 10, 20, 30, 104, 40, 40, 94, 150, 50, 10, 50, 50, 16, 10, 20, 50, 90, 10, 15, 39, 20, 20]])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z

def MKP8(x):
    #D = 105  # et 2 contraintes
    yyy = np.array([41850, 38261, 23800, 21697, 7074, 5587, 5560, 5500, 3450, 2391, 761, 460, 367, 24785, 47910, 30250, 107200, 4235, 9835, 9262, 15000, 6399, 6155, 10874, 37100, 27040, 4117, 32240, 1600, 4500, 70610, 6570, 15290, 23840, 16500, 7010, 16020, 8000, 31026, 2568, 2365, 4350, 1972, 4975, 29400, 7471, 2700, 3840, 22400, 3575, 13500, 1125, 11950, 12753, 10568, 15600, 20652, 13150, 2900, 1790, 4970, 5770, 8180, 2450, 7140, 12470, 6010, 16000, 11100, 11093, 4685, 2590, 11500, 5820, 2842, 5000, 3300, 2800, 5420, 900, 13300, 8450, 5300, 750, 1435, 2100, 7215, 2605, 2422, 5500, 8550, 2700, 540, 2550, 2450, 725, 445, 700, 1720, 2675, 220, 300, 405, 150, 70]) 
    ccc = np.array([500, 500])
    A = np.array([
        [75, 40, 365, 95, 25, 17, 125, 20, 22, 84, 75, 50, 15, 0, 0, 12, 0, 10, 0, 50, 0, 0, 10, 0, 0, 50, 60, 150, 0, 0, 75, 0, 102, 0, 0, 40, 60, 0, 165, 0, 0, 0, 45, 0, 0, 0, 25, 0, 150, 0, 0, 0, 158, 0, 85, 95, 0, 89, 20, 0, 0, 0, 0, 0, 0, 80, 0, 110, 0, 15, 0, 60, 5, 135, 0, 0, 25, 0, 300, 35, 100, 0, 0, 25, 0, 0, 225, 25, 0, 0, 0, 0, 0, 0, 0, 5, 0, 60, 0, 100, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 50, 2, 5, 5, 10, 5, 6, 11, 41, 30, 5, 40, 2, 6, 100, 10, 25, 39, 30, 13, 30, 15, 60, 5, 5, 10, 5, 15, 91, 24, 10, 15, 90, 15, 60, 5, 55, 60, 50, 75, 100, 65, 15, 10, 30, 35, 50, 15, 45, 80, 40, 110, 80, 80, 36, 20, 90, 50, 25, 50, 35, 30, 60, 10, 150, 110, 70, 10, 20, 30, 104, 40, 40, 94, 150, 50, 10, 50, 50, 16, 10, 20, 50, 90, 10, 15, 39, 20, 20]])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z    

def MKP9(x):
    # D = 60  # et 30 contraintes
    yyy = np.array([2, 77, 6, 67, 930, 3, 6, 270, 33, 13, 110, 21, 56, 974, 47, 734, 238, 75, 200, 51, 47, 63, 7, 6, 468, 72, 95, 82, 91, 83, 27, 13, 6, 76, 55, 72, 300, 6, 65, 39, 63, 61, 52, 85, 29, 640, 558, 53, 47, 25, 3, 6, 568, 6, 2, 780, 69, 31, 774, 22])  # 60 objets
    ccc = np.array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 4000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 4000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 4000])  # 30 contraintes
    A = np.array([
        [47, 774, 76, 56, 59, 22, 42, 1, 21, 760, 818, 62, 42, 36, 785, 29, 662, 49, 608, 116, 834, 57, 42, 39, 994, 690, 27, 524, 23, 96, 667, 490, 805, 46, 19, 26, 97, 71, 699, 465, 53, 26, 123, 20, 25, 450, 22, 979, 75, 96, 27, 41, 21, 81, 15, 76, 97, 646, 898, 37],
        [73, 67, 27, 99, 35, 794, 53, 378, 234, 32, 792, 97, 64, 19, 435, 712, 837, 22, 504, 332, 13, 65, 86, 29, 894, 266, 75, 16, 86, 91, 67, 445, 118, 73, 97, 370, 88, 85, 165, 268, 758, 21, 255, 81, 5, 774, 39, 377, 18, 370, 96, 61, 57, 23, 13, 164, 908, 834, 960, 87],
        [36, 42, 56, 96, 438, 49, 57, 16, 978, 9, 644, 584, 82, 550, 283, 340, 596, 788, 33, 350, 55, 59, 348, 66, 468, 983, 6, 33, 42, 96, 464, 175, 33, 97, 15, 22, 9, 554, 358, 587, 71, 23, 931, 931, 94, 798, 73, 873, 22, 39, 71, 864, 59, 82, 16, 444, 37, 475, 65, 5],
        [47, 114, 26, 668, 82, 43, 55, 55, 56, 27, 716, 7, 77, 26, 950, 320, 350, 95, 714, 789, 430, 97, 590, 32, 69, 264, 19, 51, 97, 33, 571, 388, 602, 140, 15, 85, 42, 66, 778, 936, 61, 23, 449, 973, 828, 33, 53, 297, 75, 3, 54, 27, 918, 11, 620, 13, 28, 80, 79, 3],
        [61, 720, 7, 31, 22, 82, 688, 19, 82, 654, 809, 99, 81, 97, 830, 826, 775, 72, 9, 719, 740, 860, 72, 30, 82, 112, 66, 638, 150, 13, 586, 590, 519, 2, 320, 13, 964, 754, 70, 241, 72, 12, 996, 868, 36, 91, 79, 221, 49, 690, 23, 18, 748, 408, 688, 97, 85, 777, 294, 17],
        [698, 53, 290, 3, 62, 37, 704, 810, 42, 17, 983, 11, 45, 56, 234, 389, 712, 664, 59, 15, 22, 91, 57, 784, 75, 719, 294, 978, 75, 86, 105, 227, 760, 2, 190, 3, 71, 32, 210, 678, 41, 93, 47, 581, 37, 977, 62, 503, 32, 85, 31, 36, 30, 328, 74, 31, 56, 891, 62, 97],
        [71, 37, 978, 93, 9, 23, 47, 71, 744, 9, 619, 32, 214, 31, 796, 103, 593, 16, 468, 700, 884, 67, 36, 3, 93, 71, 734, 504, 81, 53, 509, 114, 293, 31, 75, 59, 99, 11, 67, 306, 96, 218, 845, 303, 3, 319, 86, 724, 22, 838, 82, 5, 330, 58, 55, 66, 53, 916, 89, 56],
        [33, 27, 13, 57, 6, 87, 21, 12, 15, 290, 206, 420, 32, 880, 854, 417, 770, 4, 12, 952, 604, 13, 96, 910, 34, 460, 76, 16, 140, 100, 876, 622, 559, 39, 640, 59, 6, 244, 232, 513, 644, 7, 813, 624, 990, 274, 808, 372, 2, 694, 804, 39, 5, 644, 914, 484, 1, 8, 43, 92],
        [16, 36, 538, 210, 844, 520, 33, 73, 100, 284, 650, 85, 894, 2, 206, 637, 324, 318, 7, 566, 46, 818, 92, 65, 520, 721, 90, 53, 174, 43, 320, 812, 382, 16, 878, 678, 29, 92, 755, 827, 27, 218, 143, 12, 57, 480, 154, 944, 7, 730, 12, 65, 67, 39, 390, 32, 39, 318, 47, 86],
        [45, 51, 59, 21, 53, 43, 25, 7, 42, 27, 310, 45, 72, 53, 798, 304, 354, 79, 45, 44, 52, 76, 45, 26, 27, 968, 86, 16, 62, 85, 790, 208, 390, 36, 62, 83, 93, 16, 574, 150, 99, 7, 920, 860, 12, 404, 31, 560, 37, 32, 9, 62, 7, 43, 17, 77, 73, 368, 66, 82],
        [11, 51, 97, 26, 83, 426, 92, 39, 66, 2, 23, 93, 85, 660, 85, 774, 77, 77, 927, 868, 7, 554, 760, 104, 48, 202, 45, 75, 51, 55, 716, 752, 37, 95, 267, 91, 5, 956, 444, 529, 96, 99, 17, 99, 62, 7, 394, 580, 604, 89, 678, 476, 97, 234, 1, 608, 19, 69, 676, 51],
        [410, 89, 414, 81, 130, 491, 6, 238, 79, 43, 5, 288, 910, 204, 948, 19, 644, 21, 295, 11, 6, 595, 904, 67, 51, 703, 430, 95, 408, 89, 11, 495, 844, 13, 417, 570, 9, 429, 16, 939, 430, 270, 49, 72, 65, 66, 338, 994, 167, 76, 47, 211, 87, 39, 1, 570, 85, 134, 967, 12],
        [553, 63, 35, 63, 98, 402, 664, 85, 458, 834, 3, 62, 508, 7, 1, 72, 88, 45, 496, 43, 750, 222, 96, 31, 278, 184, 36, 7, 210, 55, 653, 51, 35, 37, 393, 2, 49, 884, 418, 379, 75, 338, 51, 21, 29, 95, 790, 846, 720, 71, 728, 930, 95, 1, 910, 5, 804, 5, 284, 128],
        [423, 6, 58, 36, 37, 321, 22, 26, 16, 27, 218, 530, 93, 55, 89, 71, 828, 75, 628, 67, 66, 622, 440, 91, 73, 790, 710, 59, 83, 968, 129, 632, 170, 67, 613, 608, 43, 71, 730, 910, 36, 92, 950, 138, 23, 95, 460, 62, 189, 73, 65, 943, 62, 554, 46, 318, 13, 540, 90, 53],
        [967, 654, 46, 69, 26, 769, 82, 89, 15, 87, 46, 59, 22, 840, 66, 35, 684, 57, 254, 230, 21, 586, 51, 19, 984, 156, 23, 748, 760, 65, 339, 892, 13, 13, 327, 65, 35, 246, 71, 178, 83, 3, 34, 624, 788, 200, 980, 882, 343, 550, 708, 542, 53, 72, 86, 51, 700, 524, 577, 948],
        [132, 900, 72, 51, 91, 150, 22, 110, 154, 148, 99, 75, 21, 544, 110, 11, 52, 840, 201, 2, 6, 663, 22, 20, 89, 10, 93, 964, 924, 73, 501, 398, 3, 2, 279, 5, 288, 80, 91, 132, 620, 628, 57, 79, 2, 874, 36, 497, 846, 22, 350, 866, 57, 86, 83, 178, 968, 52, 399, 628],
        [869, 26, 710, 37, 81, 89, 6, 82, 82, 56, 96, 66, 46, 13, 934, 49, 394, 72, 194, 408, 5, 541, 88, 93, 36, 398, 508, 89, 66, 16, 71, 466, 7, 95, 464, 41, 69, 130, 488, 695, 82, 39, 95, 53, 37, 200, 87, 56, 268, 71, 304, 855, 22, 564, 47, 26, 26, 370, 569, 2],
        [494, 2, 25, 61, 674, 638, 61, 59, 62, 690, 630, 86, 198, 24, 15, 650, 75, 25, 571, 338, 268, 958, 95, 898, 56, 585, 99, 83, 21, 600, 462, 940, 96, 464, 228, 93, 72, 734, 89, 287, 174, 62, 51, 73, 42, 838, 82, 515, 232, 91, 25, 47, 12, 56, 65, 734, 70, 48, 209, 71],
        [267, 290, 31, 844, 12, 570, 13, 69, 65, 848, 72, 780, 27, 96, 97, 17, 69, 274, 616, 36, 554, 236, 47, 7, 47, 134, 76, 62, 824, 55, 374, 471, 478, 504, 496, 754, 604, 923, 330, 22, 97, 6, 2, 16, 14, 958, 53, 480, 482, 93, 57, 641, 72, 75, 51, 96, 83, 47, 403, 32],
        [624, 7, 96, 45, 97, 148, 91, 3, 69, 26, 22, 45, 42, 2, 75, 76, 96, 67, 688, 2, 2, 224, 83, 69, 41, 660, 81, 89, 93, 27, 214, 458, 66, 72, 384, 59, 76, 538, 15, 840, 65, 63, 77, 33, 92, 32, 35, 832, 970, 49, 13, 8, 77, 75, 51, 95, 56, 63, 578, 47],
        [33, 62, 928, 292, 2, 340, 278, 911, 818, 770, 464, 53, 888, 55, 76, 31, 389, 40, 864, 36, 35, 37, 69, 95, 22, 648, 334, 14, 198, 42, 73, 594, 95, 32, 814, 45, 45, 515, 634, 254, 42, 29, 15, 83, 55, 176, 35, 46, 60, 296, 262, 598, 67, 644, 80, 999, 3, 727, 79, 374],
        [19, 780, 400, 588, 37, 86, 23, 583, 518, 42, 56, 1, 108, 83, 43, 720, 570, 81, 674, 25, 96, 218, 6, 69, 107, 534, 158, 56, 5, 938, 9, 938, 274, 76, 298, 9, 518, 571, 47, 175, 63, 93, 49, 94, 42, 26, 79, 50, 718, 926, 419, 810, 23, 363, 519, 339, 86, 751, 7, 86],
        [47, 75, 55, 554, 3, 800, 6, 13, 85, 65, 99, 45, 69, 73, 864, 95, 199, 924, 19, 948, 214, 3, 718, 56, 278, 1, 363, 86, 1, 22, 56, 114, 13, 53, 56, 19, 82, 88, 99, 543, 674, 704, 418, 670, 554, 282, 5, 67, 63, 466, 491, 49, 67, 154, 956, 911, 77, 635, 2, 49],
        [53, 12, 79, 481, 218, 26, 624, 954, 13, 580, 130, 608, 3, 37, 91, 78, 743, 1, 950, 45, 41, 718, 36, 30, 534, 418, 452, 359, 759, 88, 29, 499, 55, 974, 93, 56, 108, 257, 93, 171, 13, 92, 63, 714, 9, 84, 890, 16, 930, 967, 748, 5, 7, 6, 327, 894, 33, 629, 448, 21],
        [9, 19, 7, 535, 75, 3, 27, 928, 21, 7, 864, 27, 73, 61, 25, 75, 876, 16, 92, 22, 248, 11, 86, 944, 872, 996, 252, 2, 800, 334, 93, 107, 254, 441, 930, 744, 97, 177, 498, 931, 694, 800, 9, 36, 6, 539, 35, 79, 130, 860, 710, 7, 630, 475, 903, 552, 2, 45, 97, 974],
        [17, 36, 77, 843, 328, 22, 76, 368, 39, 71, 35, 850, 96, 93, 87, 56, 972, 96, 594, 864, 344, 76, 17, 17, 576, 629, 780, 640, 56, 65, 43, 196, 520, 86, 92, 31, 6, 593, 174, 569, 89, 718, 83, 8, 790, 285, 780, 62, 378, 313, 519, 2, 85, 845, 931, 731, 42, 365, 32, 33],
        [65, 59, 2, 671, 26, 364, 854, 526, 570, 630, 33, 654, 95, 41, 42, 27, 584, 17, 724, 59, 42, 26, 918, 6, 242, 356, 75, 644, 818, 168, 964, 12, 97, 178, 634, 21, 3, 586, 47, 382, 804, 89, 194, 21, 610, 168, 79, 96, 87, 266, 482, 46, 96, 969, 629, 128, 924, 812, 19, 2],
        [468, 13, 9, 120, 73, 7, 92, 99, 93, 418, 224, 22, 7, 29, 57, 33, 949, 65, 92, 898, 200, 57, 12, 31, 296, 185, 272, 91, 77, 37, 734, 911, 27, 310, 59, 33, 87, 872, 73, 79, 920, 85, 59, 72, 888, 49, 12, 79, 538, 947, 462, 444, 828, 935, 518, 894, 13, 591, 22, 920],
        [23, 93, 87, 490, 32, 63, 870, 393, 52, 23, 63, 634, 39, 83, 12, 72, 131, 69, 984, 87, 86, 99, 52, 110, 183, 704, 232, 674, 384, 47, 804, 99, 83, 81, 174, 99, 77, 708, 7, 623, 114, 1, 750, 49, 284, 492, 11, 61, 6, 449, 429, 52, 62, 482, 826, 147, 338, 911, 30, 984],
        [35, 55, 21, 264, 5, 35, 92, 128, 65, 27, 9, 52, 66, 51, 7, 47, 670, 83, 76, 7, 79, 37, 2, 46, 480, 608, 990, 53, 47, 19, 35, 518, 71, 59, 32, 87, 96, 240, 52, 310, 86, 73, 52, 31, 83, 544, 16, 15, 21, 774, 224, 7, 83, 680, 554, 310, 96, 844, 29, 61]
    ])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z

def MKP10(x):
    # D = 60  # et 30 contraintes
    yyy = np.array([2, 77, 6, 67, 930, 3, 6, 270, 33, 13, 110, 21, 56, 974, 47, 734, 238, 75, 200, 51, 47, 63, 7, 6, 468, 72, 95, 82, 91, 83, 27, 13, 6, 76, 55, 72, 300, 6, 65, 39, 63, 61, 52, 85, 29, 640, 558, 53, 47, 25, 3, 6, 568, 6, 2, 780, 69, 31, 774, 22])
    ccc = np.array([10000] * 27 + [7000] + [10000] * 2) 
    A = np.array([
        [47, 774, 76, 56, 59, 22, 42, 1, 21, 760, 818, 62, 42, 36, 785, 29, 662, 49, 608, 116, 834, 57, 42, 39, 994, 690, 27, 524, 23, 96, 667, 490, 805, 46, 19, 26, 97, 71, 699, 465, 53, 26, 123, 20, 25, 450, 22, 979, 75, 96, 27, 41, 21, 81, 15, 76, 97, 646, 898, 37],
        [73, 67, 27, 99, 35, 794, 53, 378, 234, 32, 792, 97, 64, 19, 435, 712, 837, 22, 504, 332, 13, 65, 86, 29, 894, 266, 75, 16, 86, 91, 67, 445, 118, 73, 97, 370, 88, 85, 165, 268, 758, 21, 255, 81, 5, 774, 39, 377, 18, 370, 96, 61, 57, 23, 13, 164, 908, 834, 960, 87],
        [36, 42, 56, 96, 438, 49, 57, 16, 978, 9, 644, 584, 82, 550, 283, 340, 596, 788, 33, 350, 55, 59, 348, 66, 468, 983, 6, 33, 42, 96, 464, 175, 33, 97, 15, 22, 9, 554, 358, 587, 71, 23, 931, 931, 94, 798, 73, 873, 22, 39, 71, 864, 59, 82, 16, 444, 37, 475, 65, 5],
        [47, 114, 26, 668, 82, 43, 55, 55, 56, 27, 716, 7, 77, 26, 950, 320, 350, 95, 714, 789, 430, 97, 590, 32, 69, 264, 19, 51, 97, 33, 571, 388, 602, 140, 15, 85, 42, 66, 778, 936, 61, 23, 449, 973, 828, 33, 53, 297, 75, 3, 54, 27, 918, 11, 620, 13, 28, 80, 79, 3],
        [61, 720, 7, 31, 22, 82, 688, 19, 82, 654, 809, 99, 81, 97, 830, 826, 775, 72, 9, 719, 740, 860, 72, 30, 82, 112, 66, 638, 150, 13, 586, 590, 519, 2, 320, 13, 964, 754, 70, 241, 72, 12, 996, 868, 36, 91, 79, 221, 49, 690, 23, 18, 748, 408, 688, 97, 85, 777, 294, 17],
        [698, 53, 290, 3, 62, 37, 704, 810, 42, 17, 983, 11, 45, 56, 234, 389, 712, 664, 59, 15, 22, 91, 57, 784, 75, 719, 294, 978, 75, 86, 105, 227, 760, 2, 190, 3, 71, 32, 210, 678, 41, 93, 47, 581, 37, 977, 62, 503, 32, 85, 31, 36, 30, 328, 74, 31, 56, 891, 62, 97],
        [71, 37, 978, 93, 9, 23, 47, 71, 744, 9, 619, 32, 214, 31, 796, 103, 593, 16, 468, 700, 884, 67, 36, 3, 93, 71, 734, 504, 81, 53, 509, 114, 293, 31, 75, 59, 99, 11, 67, 306, 96, 218, 845, 303, 3, 319, 86, 724, 22, 838, 82, 5, 330, 58, 55, 66, 53, 916, 89, 56],
        [33, 27, 13, 57, 6, 87, 21, 12, 15, 290, 206, 420, 32, 880, 854, 417, 770, 4, 12, 952, 604, 13, 96, 910, 34, 460, 76, 16, 140, 100, 876, 622, 559, 39, 640, 59, 6, 244, 232, 513, 644, 7, 813, 624, 990, 274, 808, 372, 2, 694, 804, 39, 5, 644, 914, 484, 1, 8, 43, 92],
        [16, 36, 538, 210, 844, 520, 33, 73, 100, 284, 650, 85, 894, 2, 206, 637, 324, 318, 7, 566, 46, 818, 92, 65, 520, 721, 90, 53, 174, 43, 320, 812, 382, 16, 878, 678, 29, 92, 755, 827, 27, 218, 143, 12, 57, 480, 154, 944, 7, 730, 12, 65, 67, 39, 390, 32, 39, 318, 47, 86],
        [45, 51, 59, 21, 53, 43, 25, 7, 42, 27, 310, 45, 72, 53, 798, 304, 354, 79, 45, 44, 52, 76, 45, 26, 27, 968, 86, 16, 62, 85, 790, 208, 390, 36, 62, 83, 93, 16, 574, 150, 99, 7, 920, 860, 12, 404, 31, 560, 37, 32, 9, 62, 7, 43, 17, 77, 73, 368, 66, 82],
        [11, 51, 97, 26, 83, 426, 92, 39, 66, 2, 23, 93, 85, 660, 85, 774, 77, 77, 927, 868, 7, 554, 760, 104, 48, 202, 45, 75, 51, 55, 716, 752, 37, 95, 267, 91, 5, 956, 444, 529, 96, 99, 17, 99, 62, 7, 394, 580, 604, 89, 678, 476, 97, 234, 1, 608, 19, 69, 676, 51],
        [410, 89, 414, 81, 130, 491, 6, 238, 79, 43, 5, 288, 910, 204, 948, 19, 644, 21, 295, 11, 6, 595, 904, 67, 51, 703, 430, 95, 408, 89, 11, 495, 844, 13, 417, 570, 9, 429, 16, 939, 430, 270, 49, 72, 65, 66, 338, 994, 167, 76, 47, 211, 87, 39, 1, 570, 85, 134, 967, 12],
        [553, 63, 35, 63, 98, 402, 664, 85, 458, 834, 3, 62, 508, 7, 1, 72, 88, 45, 496, 43, 750, 222, 96, 31, 278, 184, 36, 7, 210, 55, 653, 51, 35, 37, 393, 2, 49, 884, 418, 379, 75, 338, 51, 21, 29, 95, 790, 846, 720, 71, 728, 930, 95, 1, 910, 5, 804, 5, 284, 128],
        [423, 6, 58, 36, 37, 321, 22, 26, 16, 27, 218, 530, 93, 55, 89, 71, 828, 75, 628, 67, 66, 622, 440, 91, 73, 790, 710, 59, 83, 968, 129, 632, 170, 67, 613, 608, 43, 71, 730, 910, 36, 92, 950, 138, 23, 95, 460, 62, 189, 73, 65, 943, 62, 554, 46, 318, 13, 540, 90, 53],
        [967, 654, 46, 69, 26, 769, 82, 89, 15, 87, 46, 59, 22, 840, 66, 35, 684, 57, 254, 230, 21, 586, 51, 19, 984, 156, 23, 748, 760, 65, 339, 892, 13, 13, 327, 65, 35, 246, 71, 178, 83, 3, 34, 624, 788, 200, 980, 882, 343, 550, 708, 542, 53, 72, 86, 51, 700, 524, 577, 948],
        [132, 900, 72, 51, 91, 150, 22, 110, 154, 148, 99, 75, 21, 544, 110, 11, 52, 840, 201, 2, 6, 663, 22, 20, 89, 10, 93, 964, 924, 73, 501, 398, 3, 2, 279, 5, 288, 80, 91, 132, 620, 628, 57, 79, 2, 874, 36, 497, 846, 22, 350, 866, 57, 86, 83, 178, 968, 52, 399, 628],
        [869, 26, 710, 37, 81, 89, 6, 82, 82, 56, 96, 66, 46, 13, 934, 49, 394, 72, 194, 408, 5, 541, 88, 93, 36, 398, 508, 89, 66, 16, 71, 466, 7, 95, 464, 41, 69, 130, 488, 695, 82, 39, 95, 53, 37, 200, 87, 56, 268, 71, 304, 855, 22, 564, 47, 26, 26, 370, 569, 2],
        [494, 2, 25, 61, 674, 638, 61, 59, 62, 690, 630, 86, 198, 24, 15, 650, 75, 25, 571, 338, 268, 958, 95, 898, 56, 585, 99, 83, 21, 600, 462, 940, 96, 464, 228, 93, 72, 734, 89, 287, 174, 62, 51, 73, 42, 838, 82, 515, 232, 91, 25, 47, 12, 56, 65, 734, 70, 48, 209, 71],
        [267, 290, 31, 844, 12, 570, 13, 69, 65, 848, 72, 780, 27, 96, 97, 17, 69, 274, 616, 36, 554, 236, 47, 7, 47, 134, 76, 62, 824, 55, 374, 471, 478, 504, 496, 754, 604, 923, 330, 22, 97, 6, 2, 16, 14, 958, 53, 480, 482, 93, 57, 641, 72, 75, 51, 96, 83, 47, 403, 32],
        [624, 7, 96, 45, 97, 148, 91, 3, 69, 26, 22, 45, 42, 2, 75, 76, 96, 67, 688, 2, 2, 224, 83, 69, 41, 660, 81, 89, 93, 27, 214, 458, 66, 72, 384, 59, 76, 538, 15, 840, 65, 63, 77, 33, 92, 32, 35, 832, 970, 49, 13, 8, 77, 75, 51, 95, 56, 63, 578, 47],
        [33, 62, 928, 292, 2, 340, 278, 911, 818, 770, 464, 53, 888, 55, 76, 31, 389, 40, 864, 36, 35, 37, 69, 95, 22, 648, 334, 14, 198, 42, 73, 594, 95, 32, 814, 45, 45, 515, 634, 254, 42, 29, 15, 83, 55, 176, 35, 46, 60, 296, 262, 598, 67, 644, 80, 999, 3, 727, 79, 374],
        [19, 780, 400, 588, 37, 86, 23, 583, 518, 42, 56, 1, 108, 83, 43, 720, 570, 81, 674, 25, 96, 218, 6, 69, 107, 534, 158, 56, 5, 938, 9, 938, 274, 76, 298, 9, 518, 571, 47, 175, 63, 93, 49, 94, 42, 26, 79, 50, 718, 926, 419, 810, 23, 363, 519, 339, 86, 751, 7, 86],
        [47, 75, 55, 554, 3, 800, 6, 13, 85, 65, 99, 45, 69, 73, 864, 95, 199, 924, 19, 948, 214, 3, 718, 56, 278, 1, 363, 86, 1, 22, 56, 114, 13, 53, 56, 19, 82, 88, 99, 543, 674, 704, 418, 670, 554, 282, 5, 67, 63, 466, 491, 49, 67, 154, 956, 911, 77, 635, 2, 49],
        [53, 12, 79, 481, 218, 26, 624, 954, 13, 580, 130, 608, 3, 37, 91, 78, 743, 1, 950, 45, 41, 718, 36, 30, 534, 418, 452, 359, 759, 88, 29, 499, 55, 974, 93, 56, 108, 257, 93, 171, 13, 92, 63, 714, 9, 84, 890, 16, 930, 967, 748, 5, 7, 6, 327, 894, 33, 629, 448, 21],
        [9, 19, 7, 535, 75, 3, 27, 928, 21, 7, 864, 27, 73, 61, 25, 75, 876, 16, 92, 22, 248, 11, 86, 944, 872, 996, 252, 2, 800, 334, 93, 107, 254, 441, 930, 744, 97, 177, 498, 931, 694, 800, 9, 36, 6, 539, 35, 79, 130, 860, 710, 7, 630, 475, 903, 552, 2, 45, 97, 974],
        [17, 36, 77, 843, 328, 22, 76, 368, 39, 71, 35, 850, 96, 93, 87, 56, 972, 96, 594, 864, 344, 76, 17, 17, 576, 629, 780, 640, 56, 65, 43, 196, 520, 86, 92, 31, 6, 593, 174, 569, 89, 718, 83, 8, 790, 285, 780, 62, 378, 313, 519, 2, 85, 845, 931, 731, 42, 365, 32, 33],
        [65, 59, 2, 671, 26, 364, 854, 526, 570, 630, 33, 654, 95, 41, 42, 27, 584, 17, 724, 59, 42, 26, 918, 6, 242, 356, 75, 644, 818, 168, 964, 12, 97, 178, 634, 21, 3, 586, 47, 382, 804, 89, 194, 21, 610, 168, 79, 96, 87, 266, 482, 46, 96, 969, 629, 128, 924, 812, 19, 2],
        [468, 13, 9, 120, 73, 7, 92, 99, 93, 418, 224, 22, 7, 29, 57, 33, 949, 65, 92, 898, 200, 57, 12, 31, 296, 185, 272, 91, 77, 37, 734, 911, 27, 310, 59, 33, 87, 872, 73, 79, 920, 85, 59, 72, 888, 49, 12, 79, 538, 947, 462, 444, 828, 935, 518, 894, 13, 591, 22, 920],
        [23, 93, 87, 490, 32, 63, 870, 393, 52, 23, 63, 634, 39, 83, 12, 72, 131, 69, 984, 87, 86, 99, 52, 110, 183, 704, 232, 674, 384, 47, 804, 99, 83, 81, 174, 99, 77, 708, 7, 623, 114, 1, 750, 49, 284, 492, 11, 61, 6, 449, 429, 52, 62, 482, 826, 147, 338, 911, 30, 984],
        [35, 55, 21, 264, 5, 35, 92, 128, 65, 27, 9, 52, 66, 51, 7, 47, 670, 83, 76, 7, 79, 37, 2, 46, 480, 608, 990, 53, 47, 19, 35, 518, 71, 59, 32, 87, 96, 240, 52, 310, 86, 73, 52, 31, 83, 544, 16, 15, 21, 774, 224, 7, 83, 680, 554, 310, 96, 844, 29, 61]
    ])
    z = np.dot(x,yyy)
    weights_used = np.dot(A,x)
    z += - beta * np.sum(np.maximum(0, weights_used - ccc) )
    return - z # au lieu de z cela revient à minimizer -z
      