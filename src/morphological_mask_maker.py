

class MorphologicalMaskMaker:
    @staticmethod
    def rectangle(dy,dx):
        indicies =[]
        if dx == 0 and dy !=0:
            for j in range(-dy,dy,1):
                indicies.append([0,j])
        elif dx !=0 and dy==0:
            for i in range(-dx,dx,1):
                indicies.append([i,0])
        else:
            for i in range(-dx,dx,1):
                for j in range(-dy,dy,1):
                    indicies.append([i,j])
        return indicies