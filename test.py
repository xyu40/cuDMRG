from cuDMRG import Index, Tensor

if __name__ == "__main__":
    indices = [Index(3), Index(4), Index(5)]
    A = Tensor(indices[:-1]).setRandom()
    B = Tensor(indices[1:]).setRandom()

    print(A)
    print(B)
    print(A * B)

    print(A._data @ B._data)
