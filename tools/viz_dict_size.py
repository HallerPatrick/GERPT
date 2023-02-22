import matplotlib.pyplot as plt
import seaborn as sns

def other():

    ner_scores = [0.8789, 0.882, 0.883, 0.8877]
    upos_scores = [0.9534, 0.9564, 0.9568, 0.9568]
    class_scores = [0.8132, 0.822, 0.822, 0.8298]
    train_time = [6151, 10225, 27742, 69456] 

    dict_size = [10000, 30000, 60000, 100000]

    X = [ (2,3,1), (2,3,2), (2,3,3), (2, 1, 2) ]

    for nrows, ncols, plot_number in X:
        plt.subplot(nrows, ncols, plot_number)


    ner_plot = plt.subplot(*X[0])
    ner_plot.set_title("NER")
    ner_plot.scatter(dict_size, ner_scores, color="b")
    ner_plot.plot(dict_size, ner_scores, color="b")


    upos_plot = plt.subplot(*X[1])
    upos_plot.set_title("UPOS")
    upos_plot.scatter(dict_size, upos_scores, color="g")
    upos_plot.plot(dict_size, upos_scores, color="g")

    class_plot = plt.subplot(*X[2])
    class_plot.set_title("CLASS")
    class_plot.scatter(dict_size, class_scores, color="r")
    class_plot.plot(dict_size, class_scores, color="r")


    time_plot = plt.subplot(*X[3])
    time_plot.set_title("Training time in secs")
    time_plot.scatter(dict_size, train_time)
    time_plot.plot(dict_size, train_time)

    plt.show()


def normal_dict():

    sizes = [301, 8931, 112793, 690260]

    sns.barplot(x=list(range(1, 5)), y=sizes)
    
    plt.title("Dictionary size for $n$ on Wikitext-103")
    plt.xlabel("N-Gram")
    plt.ylabel("Number of unique tokens")
    plt.show()

if __name__ == "__main__":
    normal_dict()


