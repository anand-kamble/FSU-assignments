    
      ! ==============================================      
      ! Question 2 A
      ! ==============================================
      PROGRAM Q1_B
        INTEGER          N, NRHS, KL, KU
        PARAMETER        ( N = 10, NRHS = 1, KL = 2, KU =2)
        INTEGER          LDA, LDB
        PARAMETER        ( LDA = N, LDB = N )
        INTEGER          INFO
        INTEGER          IPIV( N )
        
        INTEGER LDAB
        PARAMETER (LDAB = 2*KL+KU+1)
        DOUBLE PRECISION A( LDAB,N), B( LDB, NRHS),  C( LDB, NRHS)

        EXTERNAL DGBTRF, DGBTRS
          
        DO i=1,LDAB
            DO J=1,N
                A(i,j) = 1.0
                IF(i == 1 .or. i == 2) THEN 
                    A(i,j)=0.0
                ELSE IF (i == 3) THEN
                    A(i,j) = 2
                ELSE IF (i == 4) THEN
                    A(i,j) = -1.0
                ELSE IF (i == 5) THEN
                    A(i,j) = 8.0
                ELSE IF (i == 6) THEN
                    A(i,j) = 1.0
                ELSE IF (i == 7) THEN
                    A(i,j) = 3.0
                END IF
                ! WRITE(*,*) A(i,j)
            END DO
        END DO

        

        WRITE(*,*)'Value of A'

        DO i = 1, LDAB
            write(*,'(1X,*(F8.3))') (A(i,j), j=1,N)
        END DO     

        WRITE(*,*)'Value of X'
        CALL DGBTRF(N, N,KL,KU,A,LDAB, IPIV,INFO)
  
        DO i = 1, LDAB
            write(*,'(1X,*(F8.3))') (A(i,j), j=1,N)
        END DO
  

  
        ! CALL DGBTRF( N, NRHS, A, LDA, IPIV, C, LDB, INFO )
        WRITE(*,*)'Values of Y'
        DO i=1,2
              WRITE(*,*) C(i,1) 
        END DO
    
        END