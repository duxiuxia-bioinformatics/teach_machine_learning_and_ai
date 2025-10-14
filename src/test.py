input_str = input()

if len(input_str) % 2 == 0:
    print(input_str[:int(len(input_str) / 2)])
    print(input_str[int(len(input_str)/2):])
else:
    print(input_str[:int((len(input_str)-1)/2)])
    print(input_str[int((len(input_str)-1)/2):])

