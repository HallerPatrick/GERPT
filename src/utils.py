
def display_text(dictionary, t):
    for a in t:
        print(repr(dictionary.idx2word[a.item()]), end="")
    print()

def display_input_n_gram_sequences(input, dictionary):
    print("Target sequences:")
    for i in range(input.size()[0]):
        print(f"{i+1}-gram")
        display_text(dictionary, input[i])
