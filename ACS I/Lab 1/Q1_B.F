    
      ! ==============================================      
      ! Question 1 B
      ! ==============================================
      PROGRAM Q1_B
      INTEGER          N, NRHS
      PARAMETER        ( N = 2, NRHS = 1 )
      INTEGER          LDA, LDB
      PARAMETER        ( LDA = N, LDB = N )
      INTEGER          INFO
      INTEGER          IPIV( N )
      DOUBLE PRECISION A( LDA, N), B( LDB, NRHS),  C( LDB, NRHS), X(1,2)
      EXTERNAL DGESV
        
      DATA  A/ 4.5, 1.6, 3.1, 1.1 /

      DATA  B/ 19.249, 6.843 /

      DATA  C/ 19.25, 6.84 /

      WRITE(*,*) 'Initialized Matrix A:'
      DO i = 1, N
        WRITE(*,*) (A(i, j), j = 1, 2)
      END DO

      WRITE(*,*) 'Initialized Matrix B:'
      DO i = 1, N
        WRITE(*,*) (B(i,1))
      END DO

      WRITE(*,*) 'Initialized Matrix C:'
      DO i = 1, N
        WRITE(*,*) (C(i,1))
      END DO



      CALL DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
      WRITE(*,*) 'Solution to Linear System 1 (X):'
      DO i = 1, N
        WRITE(*,*) B(i, 1)
      END DO
        
      DATA  A/ 4.5, 1.6, 3.1, 1.1 /
      INFO = 0

      CALL DGESV( N, NRHS, A, LDA, IPIV, C, LDB, INFO )
      WRITE(*,*)'Solution to Linear System 2 (Y):'
      DO i=1,2
            WRITE(*,*) C(i,1) 
      END DO
  
      END