from user_inference import rank_articles

a = rank_articles(["sports", "soccer", "Munich vs Dortmund"],
              [['fc bayern', 'championship', 'munich']])

print(a)