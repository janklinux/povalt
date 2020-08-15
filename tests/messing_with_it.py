from povalt.training.training import TrainPotential

binpath = TrainPotential.find_binary('lsds')

if binpath is not None:
    print(binpath)
else:
    print('not existing')
