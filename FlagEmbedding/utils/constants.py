TASK_DESCRIPTION = "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer."

# TASK_DESCRIPTION = "Given a multiple choice math question and a student's wrong answer to it, retrieve the math misconception behind the wrong answer."

RERANKER_PROMPT = 'Given a query A and a passage B, determine whether the passage B explains the mathematical misconception that leads to the wrong answer in query A by providing a prediction of either "Yes" or "No".'

# "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."