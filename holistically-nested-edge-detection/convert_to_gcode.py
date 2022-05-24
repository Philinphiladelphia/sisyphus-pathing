tablemmx = 190
tablemmy = 242
feedrate = 3600

def convertToGcode(path, maxSizeX, maxSizeY):
    gcode = ""

    for point in path:
        xLoc = (point[0]/maxSizeX)*tablemmx
        yLoc = (point[0]/maxSizeY)*tablemmy
        gcode += f"G1 X{xLoc} Y{yLoc} F{feedrate}\n"

    return gcode