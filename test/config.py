def generate_sequence(start, stop):
    current_value = start
    while current_value <= stop:
        yield current_value
        current_value *= 2

def generate_text_file(filename):
    with open(filename, 'w') as file:
        case_count = 1
        for M in generate_sequence(16, 8192):
            for N in generate_sequence(1024, 32768):
                for K in generate_sequence(1024, 32768):
                    line = f'm{M}n{N}k{K}_case{case_count}\n'
                    file.write(line)
                    case_count += 1

if __name__ == "__main__":
    generate_text_file("config.txt")
