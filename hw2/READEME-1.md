works:
 distances = []
    for i in pts:
        distances.append(np.sqrt(np.square(int(i[0])) + np.square(int(i[1]))))
    print(distances)
    distances = np.array(distances)
    indices = np.argsort(distances)
    print(indices)
    closest = []
    for i in range(k):
        closest.append(pts[indices[i]])
    print(closest)
    return closest