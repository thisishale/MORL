import numpy as np
def find_in(A, B, eps=0.2):
    # find element of A in B with a tolerance of relative err of eps.
    cnt1, cnt2 = 0.0, 0.0
    for a in A:
        for b in B:
            print(a, b)
            if eps > 0.001:
              if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                  cnt1 += 1.0
                  break
            else:
              if np.linalg.norm(a - b, ord=1) < 0.5:
                  cnt1 += 1.0
                  break
    print('*'*20)
    for b in B:
        for a in A:
            print(a, b)
            if eps > 0.001:
              if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                  cnt2 += 1.0
                  break
            else:
              if np.linalg.norm(a - b, ord=1) < 0.5:
                  cnt2 += 1.0
                  break
    return cnt1, cnt2

aa = np.array([[11,22,33],[22,33,44],[12,25,596],[1,99,56]])
bb = np.array([[11,22,33],[22,33,44],[55,1,0.5],[0.001,12,100]])
c1, c2 = find_in(aa, bb, 0.0)
print(c1)
print(c2)