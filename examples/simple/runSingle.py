
import os
import numpy as np

os.chdir("cmake-build-debug")

# Trial Params
n_steps = 1000000

# SimParams
grid_w = 60
grid_h = 60
grid_d = 60
dx = 1.0/240.0
g = -9.8
# SimWaterParams
density = 1000.0
sft = 0.00000
nu = 0.00
# SimFerroParams
M_s = 30.0*22.8*1000.0
m = 2.0*2.0E-19
kb = 1.38064852E-23
T = 273.15
interface_reg = 1.0
grid_w_em = 62
grid_h_em = 62
grid_d_em = 62
appStrength = 2000

paramNames = ["n_steps",
              # SimParams
              "grid_w",
              "grid_h",
              "grid_d",
              "dx",
              # SimWaterParams
              "density",
              "sft",
              "nu",
              "g",
              # SimFerroParams
              "M_s",
              "m",
              "kb",
              "T",
              "interface_reg",
              "grid_w_em",
              "grid_h_em",
              "grid_d_em",
              "appStrength"]


def makeParamList():
    paramListOut = [
                n_steps,
                grid_w,
                grid_h,
                grid_d,
                dx,
                density,
                sft,
                nu,
                g,                
                M_s,
                m,
                kb,
                T,
                interface_reg,
                grid_w_em,
                grid_h_em,
                grid_d_em,
                appStrength
            ]
    return paramListOut


def paramListToArgs(paramListIn):
    argsOut = ""
    for param in paramList:
        argsOut = argsOut + " "
        argsOut = argsOut + str(param)
    return argsOut


#sft_range = np.arange(0, 0.010, 0.001)


tName = "trial_single"

# Edit the parameters here for each trial

paramList = makeParamList()
args = paramListToArgs(paramList)

print(args)

os.system("./testSimFerro" + args)
os.chdir("../screens")
videoName = tName
os.system("ffmpeg -i screen_%07d.png -c:v libx264 " + videoName + ".mp4")
os.system("rm *.png")
with open(tName + ".txt", 'w') as f:
    for param, paramName in zip(paramList, paramNames):
        f.write("%s: %s\n" % (paramName, param))
os.chdir("../cmake-build-debug")
