import os
from src.evaluation import answer_mapping

if __name__ == "__main__":
    decoded = """A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?	 GT_ANS: 3	 GENERATED_ANS: Please follow the examples and answer the given question step-by-step.

                Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?; Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
                Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?; Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
                Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?; Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
                Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?; Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

                Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?; Answer step-by-step:Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?; Step-by-Step Answer: 2 bolts of blue fiber and half that much white fiber. 2 * 2 = 4 bolts of blue fiber. 2 + 4 = 6 bolts of white fiber. 6 bolts of white fiber. The answer is 6.
                Question: A bag contains 4 apples and 6 oranges. If 2 apples and 3 oranges are taken out, how many apples and oranges are left?; Answer step-by-step: Question"""
    print(answer_mapping(decoded, 4))
