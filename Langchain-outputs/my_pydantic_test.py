from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class Student(BaseModel):
    name: str = "rajab"  # for default values, use equalto sign
    age: Optional[int]
    email: EmailStr
    cgpa: float = Field(
        gt=0, lt=4, default=2.8, description="University grade of student"
    )


stu_data = {"age": 18, "email": "abc@exp.com", "cgpa": 3.0}

new_student = Student(**stu_data)

print(new_student)
print(type(new_student))

student_dict = dict(new_student)
print(f"student age = {student_dict['age']}")
student_json = new_student.model_dump_json()
print(student_json)
# pydantic use strict validation, have tons of useful features. basically works for typesafety
