import tensorflow as tf
import numpy as np

def create_tensor_const():
    tensor = tf.constant([[23, 4], [32, 51]])
    print(f"Cria constantes como tensor: {tensor}")
    print(f"Shape: {tensor.shape}")

    numpy_tensor = np.array([[23, 4], [32, 51]])
    print(f"Cria constantes a partir do numpy: {tf.constant(numpy_tensor)}")
    print(f"Shape {tf.constant(numpy_tensor).shape}")

def create_tensor(*args):
    tensor = tf.Variable(np.array(args))
    print(f"Cria variáveis como tensor: {tensor}")
    print(f"Shape: {tensor.shape}")

    return tensor

def update_tensor(tensor_position, new_value):
    tensor = tensor_position.assign(new_value)
    print(f"Atualiza variáveis como tensor: {tensor}")
    print(f"Shape: {tensor.shape}")

def square(tensor):
    square_tensor = np.square(tensor)
    print(f"O quadrado do tensor: {square_tensor}")

def square_root(tensor):
    square_root_tensor = np.sqrt(tensor)
    print(f"A raíz quadrada do tensor: {square_root_tensor}")

def matmul(tensor_one, tensor_two):
    tensor = tf.linalg.matmul(tensor_one, tensor_two, transpose_b=True)
    print(f"Multiplicação dos tensores: {tensor}")
    print(f"Shape: {tensor.shape}")

    return tensor

def main():
    tensor_one = create_tensor([1, 2, 3], [4, 5, 6])
    update_tensor(tensor_one[0, 2], 100)

    square(tensor_one)
    square_root(tensor_one)

    tensor_two = create_tensor([7, 8, 9], [0, 1, 0])

    tensor_matmul = matmul(tensor_one, tensor_two)

if __name__ == '__main__':
    main()