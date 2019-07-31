from utilities import genXdis

x = genXdis(100, 100, 10, type="Bern", prob=0.1)
print((x.sum())/100/100/10)
