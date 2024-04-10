import random as rand

def random_serialize(expression, random = True):

    expression = expression.replace(" ", "")
    serialized_exp = random_serialize_helper(expression, 0, len(expression) -1, random)

    spaced_exp = " ".join(serialized_exp)

    return spaced_exp

def random_serialize_helper(expression, start, end, random):

    closed_index = find_closed_parenthesis(expression, start)
    # print("closed index = {}".format(closed_index))

    if expression[start] == '(' and closed_index == end:
        return random_serialize_helper(expression, start+ 1, end -1, random)

    sign_positions = find_sign_positions(expression, start, end)
    # print("sign positions = {}".format(sign_positions))

    if len(sign_positions) == 0:
        sub = expression[start:end +1]
        return [sub]
    else:
        #randomly pick one sign and do the partition along there

        sign_pos = sign_positions[0]

        if random:
            num_positions = len(sign_positions)
            idx = rand.randint(0, num_positions - 1)
            sign_pos = sign_positions[idx]

        # print(num_positions, idx, sign_pos)
    
        sign = expression[sign_pos]
        sub1 = random_serialize_helper(expression, start, sign_pos -1, random)
        sub2 = random_serialize_helper(expression, sign_pos + 1, end, random)

        if random:
            coin_flip = rand.randint(0,1)
            
            if coin_flip == 1 and sign in "+*": 
                return [sign] + sub2 + sub1
            else:
                return [sign] + sub1 + sub2

        return [sign] + sub1 + sub2
    
def find_closed_parenthesis(expression, start):

    open_count = 0

    if expression[start] != '(':
        return -1

    for i in range(start + 1, len(expression)):

        if expression[i] == '(':
            open_count += 1
        elif expression[i] == ')' and open_count == 0:
            return i
        elif expression[i] == ')':
            open_count -= 1

    return -1

#This find sign will need to turn into find viable signs
#need to update this method to work on expressions with parenthesis

def find_sign_positions(expression, start, end):

    sign_priority = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3, '_': 4, '=': 0}
    sign_pos_list = []
    curr_priority = 5
    signs = "+-*/^_="

    i = start

    if expression[i] == '-':
        i += 1

    while i < end + 1:
        if expression[i] == '(':
            closed = find_closed_parenthesis(expression, i)
            i = closed
        elif expression[i] in signs and sign_priority[expression[i]] < curr_priority:
            sign_pos_list = [i]
            curr_priority = sign_priority[expression[i]]
        elif expression[i] in signs and sign_priority[expression[i]] == curr_priority:
            sign_pos_list.append(i)

        i += 1

    return sign_pos_list

def unserialize(expression):

    expression_list = expression.split()

    big_tree_end = find_tree_end(expression_list, 0)
    print(big_tree_end)

    if big_tree_end != len(expression_list) - 1:
        print("Expression is invalid!")

    unserialized_expression = unserialize_helper(expression_list, 0, len(expression_list) -1)

    return unserialized_expression


def unserialize_helper(expression_list, start, end):

    if start == end and expression_list[start] not in "*/+-^_=":
        return expression_list[start]

    if expression_list[start] in "*/+-^_=":
        tree_end = find_tree_end(expression_list, start + 1)

        left_expression = unserialize_helper(expression_list, start + 1, tree_end)
        right_expression = unserialize_helper(expression_list, tree_end + 1, end)
        sign = expression_list[start]

        return "({} {} {})".format(left_expression, sign,right_expression)

    
def find_tree_end(expression_list, start):

    if expression_list[start] not in "*/+-^_=":#if the start is just a number, the tree end is the index itself
        return start

    open_count = 2

    for i in range(start + 1, len(expression_list)):

        if expression_list[i] not in "*/+-^_=":
            open_count -= 1
        else:
            open_count += 1

        if open_count == 0:
            return i

    return -1