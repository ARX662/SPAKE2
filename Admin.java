import uk.ac.ic.doc.jpair.pairing.BigInt;
import uk.ac.ic.doc.jpair.pairing.Point;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Random;
public class Admin {
    public static void main(String[] args) throws Exception {

////////////////////////////////////////////////////////////////////////////// Admin
        // waiting for client to send data
        try {
            // Sleep for 10 seconds
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            // Do nothing
        }

        // reads Password from the user

        BufferedReader passwordreader = new BufferedReader(new FileReader("Password.txt"));
        String passwordB = passwordreader.readLine();
        passwordreader.close();
        BigInt wB = new BigInt(passwordB);

        String wBS = wB.toString();


        // N will be another point used for calculating  of S which will be sent to Alice
        BigInt n = new BigInt(160,new Random());

        Point N = Generators.g1.multiply(Generators.P,n);
        //Creating a sub Generators wN and send to client
        Point wN = Generators.g1.multiply(N, wB);



        // g^b = bP
        //Point aP is computed over G1
        BigInt b = new BigInt(160, new Random());
        Point Y = Generators.g1.multiply(Generators.P, b);

        //SB will be sent to alice
        Point SB = Generators.g1.add(Y,wN);
        String SSB = SB.toString();


        //send  data to client
        BigInt xB = (BigInt) SB.getX();
        BigInt yB = (BigInt) SB.getY();
        String xStringB = xB.toString();
        String yStringB = yB.toString();
        String pointStringS = xStringB + "," + yStringB;
        FileOutputStream fosS = new FileOutputStream("S.txt");
        DataOutputStream outStreamS = new DataOutputStream(new BufferedOutputStream(fosS));
        outStreamS.write(pointStringS.getBytes(StandardCharsets.UTF_8));
        outStreamS.close();
        // send the Generators to client
        BigInt Nx = (BigInt) wN.getX();
        BigInt Ny = (BigInt) wN.getY();
        String NStringx = Nx.toString();
        String NStringy = Ny.toString();
        String pointStringN = NStringx + "," + NStringy;
        FileOutputStream fosN = new FileOutputStream("wN.txt");
        DataOutputStream outStreamN = new DataOutputStream(new BufferedOutputStream(fosN));
        outStreamN.write(pointStringN.getBytes(StandardCharsets.UTF_8));
        outStreamN.close();
        // Encrypt file
        File S_text = new File("S.txt");
        File wN_text = new File("wN.txt");
        AESEncryption.encrypt(wB.toByteArray(),S_text);
        AESEncryption.encrypt(wB.toByteArray(),wN_text);
        // waiting for client to send data
        try {
            // Sleep for 5 seconds
            Thread.sleep(7500);
        } catch (InterruptedException e) {
            // Do nothing
        }

        // decrypt
        File T_decrypt = new File("T.txt");
        File wM_decrypt = new File("wM.txt");
        AESEncryption.decrypt(wB.toByteArray(),T_decrypt);
        AESEncryption.decrypt(wB.toByteArray(),wM_decrypt);

        // read data from client
        BufferedReader readerT = new BufferedReader(new FileReader("T.txt"));
        String StringT = readerT.readLine();
        readerT.close();

        String [] coordinatesB = StringT.split(",");
        BigInt xR = new BigInt(coordinatesB[0]);
        BigInt yR = new BigInt(coordinatesB[1]);
        Point TB = new Point(xR, yR);
        String TSB =  TB.toString();

        // read the sub Generators from client
        BufferedReader readerM = new BufferedReader(new FileReader("wM.txt"));
        String StringwM = readerM.readLine();
        readerM.close();

        String [] coordinatesM = StringwM.split(",");
        BigInt xM = new BigInt(coordinatesM[0]);
        BigInt yM = new BigInt(coordinatesM[1]);
        Point wM = new Point(xM,yM);



        //Bob will compute the following : a(T-wM) =abP
        Point X = Generators.g1.subtract(TB,wM);
        Point Kb = Generators.g1.multiply(X,b);
        //Converting to String
        String KbS = Kb.toString();


        // hash of values sent and available to Bob(Hashing)

        String FinalStringBob = TSB+SSB+KbS+wBS;
        MessageDigest messageB = MessageDigest.getInstance("SHA-256");
        byte[] hashB = messageB.digest(FinalStringBob.getBytes(StandardCharsets.UTF_8));
        BigInt sB = new BigInt(1, hashB);
        String Sb = sB.toString();

        // saving the Session key for Client to check
        FileOutputStream Session = new FileOutputStream("session_keyAdmin.txt");
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Session));
        out.write(Sb.getBytes(StandardCharsets.UTF_8));
        out.close();

        // waiting for client to send session key
        try {
            // Sleep for 10 seconds
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            // Do nothing
        }
        BufferedReader session = new BufferedReader(new FileReader("session_key.txt"));
        String SA = session.readLine();
        // session key
        BigInt sA = new BigInt(SA);

        session.close();

        if(sB.equals(sA)){
            System.out.println("Admin:Session Secured");

        }
        else{
            System.out.println("Session Incomplete!");

        }


    }




}


