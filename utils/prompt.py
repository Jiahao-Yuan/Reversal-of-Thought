reversal_of_thought='''
###Instruction###
 You are a highly distinguished expert in mathematics and information reasoning. Based on the given example, define the spe
cific task, including the task definition, pseudocode, logical pseudocode, case examples, and input-output format.
 1. Understand Task Description:
 Meticulously study demonstrations to deeply understand generic task description.
 2. Plan Generic Pseudocode:
 Provide pseudocode in text form and plan an efficient algorithm to complete the task with your experiences.
 3. Formulate Logical Pseudocode:
 Convert the pseudocode into generic logical algorithm pseudocode using ONLY logical symbols:
 Logical Operators:
 Conjunction: A ∧ B ; Disjunction: A ∨ B
 equivalence: A ≡ B , Negation: ¬A
 Quantifiers:
 Universal quantifier: ∀x ; Existential quantifier: ∃x
 Inequalities:
 Less than: x < y ; Greater than: x > y 
 Less than or equal to: x ≤ y
 Greater than or equal to: x ≥ y
 Equals: x = y ; Not equals: x= y
 Conditional Statements:
 If A then B: A ⊃ B
 If A ∧B then C: (A∧B) ⊃ C
 If A ∨B then C: (A∨B) ⊃ C
 If ∀x(P(x)) then Q: ∀x(P(x)) ⊃ Q
 If ∃x(P(x)) then Q: ∃x(P(x)) ⊃ Q etc.
 Input: [Demonstration] Output: [Output]
'''
Pair_pre='''
Please choose your more preferred instruction: A or B ?
Input:
'''
