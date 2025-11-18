your_dictionary = {}
new_key = input('Type a key')
new_value = input('Type a value')
your_dictionary[new_key] = new_value
new_key = input('Type a key')
while new_key != 'STOP':
    if new_key in your_dictionary:
        new_key = input(f'{new_key} already exists. Type a different key')
        continue

    value = input('Type a value')
    your_dictionary[new_key] = value
    new_key = input('Type a key')
print(your_dictionary)