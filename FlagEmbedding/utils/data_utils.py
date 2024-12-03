import pandas as pd

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def preprocess_data(train_data, 
                    misconception_mapping,
                    with_misconception=True, 
                    filter_na_misconception=True):

    # 1. Melt answer columns and create base dataframe
    answer_cols = ['AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']
    answer_values = ['A', 'B', 'C', 'D']

    # Melt the answer columns
    melted_answers = pd.melt(
        train_data,
        id_vars=['QuestionId', 'QuestionText', 'ConstructId', 'ConstructName', 
                'SubjectId', 'SubjectName', 'CorrectAnswer'],
        value_vars=answer_cols,
        var_name='AnswerColumn',
        value_name='WrongAnswerText'
    )
    # Add WrongAnswer column based on AnswerColumn
    melted_answers['WrongAnswer'] = melted_answers['AnswerColumn'].map(
        dict(zip(answer_cols, answer_values))
    )


    # 2. Add MisconceptionId and MisconceptionName if with_misconception = True
    if with_misconception:
        misconception_cols = [f'Misconception{x}Id' for x in ['A', 'B', 'C', 'D']]  # Fixed column names
        melted_misconceptions = pd.melt(
            train_data,
            id_vars=['QuestionId', 'CorrectAnswer'],
            value_vars=misconception_cols,
            var_name='MisconceptionColumn',
            value_name='MisconceptionId'
        )
        melted_misconceptions['WrongAnswer'] = melted_misconceptions['MisconceptionColumn'].str[-3]
        
        df = melted_answers.merge(
            melted_misconceptions[['QuestionId', 'WrongAnswer', 'MisconceptionId']], 
            on=['QuestionId', 'WrongAnswer'], 
            how='left'
        )

        df = df.merge(
            misconception_mapping[['MisconceptionId', 'MisconceptionName']], 
            on='MisconceptionId', 
            how='left'
        )
    else:
        df = melted_answers

    # Create CorrectAnswerText column
    correct_answers = df[['QuestionId', 'WrongAnswer', 'WrongAnswerText']].copy()
    correct_answers = correct_answers[
        correct_answers['WrongAnswer'] == correct_answers['QuestionId'].map(
            train_data.set_index('QuestionId')['CorrectAnswer']
        )
    ]
    correct_answers = correct_answers.rename(
        columns={'WrongAnswerText': 'CorrectAnswerText'}
    )[['QuestionId', 'CorrectAnswerText']]
    # Merge correct answer text
    df = df.merge(correct_answers, on='QuestionId', how='left')
    # Filter out the correct answer
    df = df[df['WrongAnswer'] != df['CorrectAnswer']]
    # Create QuestionId_Answer column
    df['QuestionId_Answer'] = df['QuestionId'].astype(str) + '_' + df['WrongAnswer']
    if with_misconception:
        final_columns = ['QuestionId_Answer', 'QuestionId', 'QuestionText', 'ConstructId',
            'ConstructName', 'SubjectId', 'SubjectName', 'CorrectAnswer', 'CorrectAnswerText',
            'WrongAnswerText', 'WrongAnswer', 'MisconceptionId', 'MisconceptionName']
    else:
        final_columns = ['QuestionId_Answer', 'QuestionId', 'QuestionText', 'ConstructId',
            'ConstructName', 'SubjectId', 'SubjectName', 'CorrectAnswer', 'CorrectAnswerText',
            'WrongAnswerText', 'WrongAnswer']
    df = df[final_columns]
    for col in ['QuestionText', 'ConstructName', 'SubjectName', 'CorrectAnswerText', 'WrongAnswerText']:
        df[col] = df[col].str.strip()
    # filter out rows with NA in MisconceptionId
    if with_misconception and filter_na_misconception:
        df = df[df['MisconceptionId'].notna()]
    
    df = df.sort_values(['QuestionId', 'QuestionId_Answer']).reset_index(drop=True)
    
    return df