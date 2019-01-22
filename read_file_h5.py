import h5py

filename = 'weights_hourglass.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
print(f[a_group_key])
print(list(f[a_group_key]))
