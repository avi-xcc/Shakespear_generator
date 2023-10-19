


with open("data/input.txt") as f:
    dataset = f.read()

dataset = dataset.lower()
all_unique_letters = sorted(set(dataset))

# This is done so that 0th index has space
all_unique_letters[0] = ' '
all_unique_letters[1] = '\n'

letter_to_number = {letter: numb for numb, letter in enumerate(all_unique_letters)}
number_to_letter = {numb: letter for numb, letter in enumerate(all_unique_letters)}

encode = lambda s: [letter_to_number[c] for c in s]
decode = lambda n: ''.join([number_to_letter[ni] for ni in n])


# sentence = "This is shakespeare"
#
# print(encode(sentence.lower()))
# print(list(range(len(sentence))))


