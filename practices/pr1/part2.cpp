#include <iostream>
#include <omp.h>

struct Node {
    int value;
    Node* next;
};

class List {
    Node* head;
public:
    List() : head(nullptr) {}

    void push(int x) {
        Node* n = new Node{x, head};
        head = n;
    }

    void push_parallel(int x) {
        #pragma omp critical
        push(x);
    }

    bool find(int x) {
        Node* p = head;
        while (p) {
            if (p->value == x) return true;
            p = p->next;
        }
        return false;
    }
};

class Stack {
    Node* top;
public:
    Stack() : top(nullptr) {}

    void push(int x) {
        top = new Node{x, top};
    }

    int pop() {
        int v = top->value;
        Node* t = top;
        top = top->next;
        delete t;
        return v;
    }

    bool empty() {
        return top == nullptr;
    }
};

class Queue {
    Node* head;
    Node* tail;
public:
    Queue() : head(nullptr), tail(nullptr) {}

    void push(int x) {
        Node* n = new Node{x, nullptr};
        if (!tail) head = tail = n;
        else {
            tail->next = n;
            tail = n;
        }
    }

    void push_parallel(int x) {
        #pragma omp critical
        push(x);
    }

    int pop() {
        int v = head->value;
        Node* t = head;
        head = head->next;
        if (!head) tail = nullptr;
        delete t;
        return v;
    }
};

int main() {
    List list;
    list.push(10);
    list.push(20);
    list.push(30);

    std::cout << "Find 20 = " << list.find(20) << "\n";

    Stack st;
    st.push(1);
    st.push(2);
    std::cout << "Stack pop = " << st.pop() << "\n";

    Queue q;
    q.push(100);
    q.push(200);
    std::cout << "Queue pop = " << q.pop() << "\n";

    int N;
    std::cout << "Enter number of elements: ";
    std::cin >> N;

    Queue q2;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        q2.push_parallel(i);
    }

    std::cout << "Parallel insert finished\n";

    return 0;
}
