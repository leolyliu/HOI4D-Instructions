import os


def prepare_datalists(root_dir):
    datalist = []
    for ZY in os.listdir(root_dir):
        p1 = os.path.join(root_dir, ZY)
        if not os.path.isdir(p1):
            continue
        for H in os.listdir(p1):
            p2 = os.path.join(p1, H)
            if not os.path.isdir(p2):
                continue
            for C in os.listdir(p2):
                p3 = os.path.join(p2, C)
                if not os.path.isdir(p3):
                    continue
                for N in os.listdir(p3):
                    p4 = os.path.join(p3, N)
                    if not os.path.isdir(p4):
                        continue
                    for S in os.listdir(p4):
                        p5 = os.path.join(p4, S)
                        if not os.path.isdir(p5):
                            continue
                        for s in os.listdir(p5):
                            p6 = os.path.join(p5, s)
                            if not os.path.isdir(p6):
                                continue
                            for T in os.listdir(p6):
                                p7 = os.path.join(p6, T)
                                if not os.path.isfile(os.path.join(p7, "3Dseg", "label.txt")):
                                    continue
                            datalist.append(os.path.join(ZY, H, C, N, S, s, T))
    return datalist


if __name__ == "__main__":
    root_dir = "/share/datasets/HOI4D_overall"
    datalist = prepare_datalists(root_dir)
    f = open("./datalists/train_all.txt", "w")
    for d in datalist:
        f.write(d + "\n")
    f.close()
