a = 200
b = 33
if b > a:
  print("b is greater than a")
# elif: if the previous conditions were not true, then try this condition
elif a == b:
  print("a and b are equal")
# catches anything which isn't caught by the preceding conditions
else:
  print("a is greater than b")