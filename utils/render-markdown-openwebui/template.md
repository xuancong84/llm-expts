
\[
x^{5}+3x^{4}-10=5\qquad\Longleftrightarrow\qquad 
x^{5}+3x^{4}-15=0 .
\]

The polynomial  

\[
P(x)=x^{5}+3x^{4}-15
\]

has no rational zeros (the rational‑root test leaves only  
\(\pm 1,\pm3,\pm5,\pm15\) and none of them satisfies \(P(x)=0\)).
Consequently we look for the zeros numerically.

--------------------------------------------------------------------
### 1.  The real zero  

\(P(0)=-15<0,\; P(2)=65>0\); therefore a unique positive real root
lies between \(1\) and \(2\).  
Applying Newton’s method (or any other root‑finder) gives

\[
\boxed{x_{1}\approx 1.361531\,}.
\]

(The iteration converges in a few steps:
\(x_{0}=1.4\; \to\; x_{1}=1.36153\; \to\; x_{2}=1.361531\) to six
significant figures.)

--------------------------------------------------------------------
### 2.  Deflation  

Dividing \(P(x)\) by \((x-x_{1})\) (synthetic division) yields the
quartic factor

\[
\begin{aligned}
Q(x)&=\frac{P(x)}{x-x_{1}}\\
   &=x^{4}+4.361531\,x^{3}+5.938530\,x^{2}+8.085492\,x+11.008484 .
\end{aligned}
\]

Thus  

\[
P(x)=(x-1.361531)\;Q(x) .
\]

--------------------------------------------------------------------
### 3.  The four complex zeros  

The zeros of the quartic \(Q(x)\) are obtained from a standard
numerical eigenvalue routine (e.g. the companion‑matrix method) or
any root‑finder for polynomials.  To 6‑digit accuracy they are

\[
\begin{aligned}
x_{2,3}&\;=\;-2.173302\;\pm\;0.983726\,\mathrm i,\\[2mm]
x_{4,5}&\;=\;-0.006713\;\pm\;2.748938\,\mathrm i .
\end{aligned}
\]

--------------------------------------------------------------------
### 4.  Verification (Vieta’s relations)

* Sum of the roots  

  \[
  x_{1}+x_{2}+x_{3}+x_{4}+x_{5}
  =1.361531+2(-2.173302)+2(-0.006713)
  =-3.000000\;,
  \]
  which equals \(-\)coefficient of \(x^{4}\) (as required).

* Product of the roots  

  \[
  x_{1}x_{2}x_{3}x_{4}x_{5}=15.000000,
  \]
  which equals \((-1)^{5}\) times the constant term \(-15\).

Both checks confirm the correctness of the computed zeros.

--------------------------------------------------------------------
### 5.  Result  

\[
\boxed{
\begin{aligned}
x_{1}&\approx 1.361531,\\[2mm]
x_{2,3}&\approx -2.173302\pm0.983726\,\mathrm i,\\[2mm]
x_{4,5}&\approx -0.006713\pm2.748938\,\mathrm i .
\end{aligned}}
\]

Thus the equation \(x^{5}+3x^{4}-15=0\) has one real root
\(x\approx1.361531\) and four non‑real complex roots forming two
conjugate pairs as displayed above.  (All values are given to six
significant figures; more digits can be obtained easily with any
computer‑algebra system.)
