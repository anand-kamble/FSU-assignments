
      ! ==============================================      
      ! Question 1 A
      ! ==============================================
      INTEGER          N, NRHS
      PARAMETER        ( N = 100, NRHS = 1 )
      INTEGER          LDA, LDB
      PARAMETER        ( LDA = N, LDB = N )
      INTEGER          INFO
      INTEGER          IPIV( N )
      DOUBLE PRECISION A( LDA, N), B( LDB, NRHS), X(100,1)

      
      EXTERNAL         DGESV
      WRITE(*,*)'DGESV Example Program Results'

      DO i=1,N 
            DO j=1 ,N
                  IF(i==j ) THEN 
                        A(i,j)=1.0
                  ELSE IF (j==N) THEN
                        A(i,j) = 1.0
                  ELSE IF (j > i) THEN
                        A(i,j) = -10.0
                  ELSE 
                        A(i,j) = 0.0
                  END IF
            END DO
      END DO

      DO i=1,N 
            X(i,1) = (dble(i) / (N))
      END DO

      B = MATMUL(A,X)

      
      CALL DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
      
      DO i=1,N 
            WRITE(*,*) B(i,1) 
      END DO

      END
