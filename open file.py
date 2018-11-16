# Version 1
f = open ('/Users/macuser/Desktop/Summary.txt', 'r')
message = f.read()
print(message)
f.close()

# Version 2
with open ("/Users/macuser/Desktop/Summary copy.txt") as f:
    data = f.read()
    print(data)
    f.close()