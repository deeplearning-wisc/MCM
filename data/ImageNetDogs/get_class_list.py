import json

with open('dog_tree.json') as f:
    dogs = json.load(f)
    q = [dogs]
    dogs_list = []
    while len(q) != 0:
        node = q.pop()
        if ('children' not in node):
            dogs_list.append(node['id'])
        else:
            q += node['children']
    print(str(len(dogs_list)) + ' dogs found')

with open('class_list.txt', 'w') as f:
    f.writelines('\n'.join(dogs_list))
