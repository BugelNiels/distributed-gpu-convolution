#ifndef CUSTOM_STACK_H
#define CUSTOM_STACK_H

#include <stdint.h>
extern "C" {
#include "../../util/util.h"
}

/**
 * @brief Simple lightweight stack implementation. We cannot make full use of all C++ features due to the linking phase,
 * so we provide our own implementation here. The implementation is not fully generalized, since it takes some
 * shortcuts.
 *
 * @tparam T Type of the elements in the stack.
 */
template <typename T>
struct Stack {
  int size;
  int top;
  T *elems;
};

/**
 * @brief Allocates a new empty stack.
 *
 * @tparam T Type of the elements in the stack.
 * @param size Size of the stack to allocate.
 * @return Stack<T> A newly allocated stack.
 */
template <typename T>
Stack<T> allocateStack(int size) {
  Stack<T> stack;
  stack.size = size;
  stack.top = 0;
  stack.elems = (T *)malloc(size * sizeof(T));
  return stack;
}

/**
 * @brief Pushes a new item to the stack.
 *
 * @tparam T Type of the elements in the stack.
 * @param stack The stack to push the item to.
 * @param item The item to push into the stack.
 */
template <typename T>
void push(Stack<T> *stack, T item) {
  stack->elems[stack->top] = item;
  stack->top = stack->top + 1;
}

/**
 * @brief Pops an item from the stack.
 *
 * @tparam T Type of the elements in the stack.
 * @param stack The stack to pop the item from.
 * @return T The item from the top of the stack.
 */
template <typename T>
T pop(Stack<T> *stack) {
  if (stack->top == 0) {
    // Of course not ideal, but for the current implementation, this is never meant to happen.
    FATAL("Attempting to pop from empty stack.\n");
  }
  stack->top = stack->top - 1;
  T elem = stack->elems[stack->top];
  return elem;
}

/**
 * @brief Checks whether the given stack is empty.
 *
 * @tparam T Type of the elements in the stack.
 * @param stack The stack to check.
 * @return true When there are no items in the stack.
 * @return false When the number of items in the stack is greater than 0.
 */
template <typename T>
bool isEmpty(Stack<T> *stack) {
  return stack->top == 0;
}

/**
 * @brief Get the number of items in the stack.
 *
 * @tparam T Type of the elements in the stack.
 * @param stack The stack to get the number of items of.
 * @return int Number of items currently in the stack.
 */
template <typename T>
int getSize(Stack<T> *stack) {
  return stack->top;
}

/**
 * @brief Frees the memory used by the stack.
 *
 * @tparam T Type of the elements in the stack.
 * @param stack The stack to free.
 */
template <typename T>
void freeStack(Stack<T> *stack) {
  free(stack->elems);
}

#endif  // CUSTOM_STACK_H