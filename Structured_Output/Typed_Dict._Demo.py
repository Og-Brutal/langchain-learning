from typing import TypedDict

class student(TypedDict):
    name: str
    age: int
    grade: str


student1: student={
    "name": "John Doe",
    "age": 20,
    "grade": "A"
}
print(student1)
print(student1["name"])
print(student1["age"])
print(student1["grade"])