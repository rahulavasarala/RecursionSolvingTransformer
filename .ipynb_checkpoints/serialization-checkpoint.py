def serialize(expression):

    expression = expression.replace(" ", "")

    serialized_exp = serialize_helper(expression, 0, len(expression) -1 )

    spaced_exp = " ".join(serialized_exp)

    return spaced_exp
    
def serialize_helper(expression, start, end):

    if expression[start] == '(':

        closed_index = find_closed_parenthesis(expression, start)
        if closed_index == end: #This means that there are no signs
            return serialize_helper(expression, start+ 1, end -1)
        elif closed_index == -1:
            raise Exception("No matching parenthesis found!")
        else:
            sign_index = find_sign(expression, closed_index + 1, end)
            sub1 = serialize_helper(expression, start, sign_index -1)
            sign = "" + expression[sign_index]
            sub2 = serialize_helper(expression, sign_index + 1, end)
            return [sign] + sub1 + sub2
    else:#If the expression starts with a number
        sign_index = find_sign(expression, start + 1, end)
        if sign_index == -1:
            sub = expression[start:end + 1]
            return [sub]
        else:
            sub1 = serialize_helper(expression, start, sign_index -1)
            sign = "" + expression[sign_index]
            sub2 = serialize_helper(expression, sign_index + 1, end)
            return [sign] + sub1 + sub2 
            

def find_closed_parenthesis(expression, start):

    open_count = 0

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

def find_sign(expression, start, end):

    for i in range(start, end + 1):

        if expression[i] == '(':
            break
        elif expression[i] in "+-":
            return i

    for i in range(start, end + 1):
        if expression[i] == '(':
            break
        elif expression[i] in "*/":
            return i

    for i in range(start, end + 1):
        if expression[i] == '(':
            break
        elif expression[i] in "^":
            return i

    return -1 #-1 will never be returned if closed index != end

def find_sign_positions(expression, start, end):

    sign_priority = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3, '_': 4, '=': 0}
    sign_pos_list = []
    curr_priority = 5
    signs = "+-*/^_="

    i = start

    while i < end + 1:
        print(i)
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

    if big_tree_end != len(expression_list) - 1:
        print("Expression is invalid!")

    unserialized_expression = unserialize_helper(expression_list, 0, len(expression_list) -1)\

    return unserialized_expression


def unserialize_helper(expression_list, start, end):

    if start == end and expression_list[start] not in "*/+-^":
        return expression_list[start]

    if expression_list[start] in "*/+-^":
        tree_end = find_tree_end(expression_list, start + 1)

        left_expression = unserialize_helper(expression_list, start + 1, tree_end)
        right_expression = unserialize_helper(expression_list, tree_end + 1, end)
        sign = expression_list[start]

        return "({} {} {})".format(left_expression, sign,right_expression)

    
def find_tree_end(expression_list, start):

    if expression_list[start] not in "*/+-^":#if the start is just a number, the tree end is the index itself
        return start

    open_count = 2

    for i in range(start + 1, len(expression_list)):

        if expression_list[i] not in "*/+-^":
            open_count -= 1
        else:
            open_count += 1

        if open_count == 0:
            return i

    return -1