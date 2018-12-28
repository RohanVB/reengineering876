from subprocess import call
for i in range(1, 15):
    call(["prospector", str(i) + ".py", ">Outputs/prospector" + str(i) + ".txt"])
    continue
