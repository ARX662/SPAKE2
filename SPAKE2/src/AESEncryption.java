import javax.crypto.*;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.io.*;
import java.security.*;

class AESEncryption  {



    private static final String ALGORITHM = "AES";
    private static final String TRANSFORMATION = "AES/CBC/PKCS5Padding";
    private static final int IV_SIZE = 16; // bits
    private static SecretKeySpec secretKey;

    public static void encrypt( File inputFile) throws Exception {
        SecretKeySpec key = createKey();
        byte[] iv = new byte[IV_SIZE];
        IvParameterSpec ivSpec = new IvParameterSpec(iv);

        Cipher cipher = Cipher.getInstance(TRANSFORMATION, "SunJCE");
        cipher.init(Cipher.ENCRYPT_MODE, key, ivSpec);

        FileInputStream inputStream = new FileInputStream(inputFile);
        byte[] inputBytes = new byte[(int) inputFile.length()];
        inputStream.read(inputBytes);

        byte[] outputBytes = cipher.doFinal(inputBytes);

        FileOutputStream outputStream = new FileOutputStream(inputFile);
        DataOutputStream outputStreamE = new DataOutputStream(new BufferedOutputStream(outputStream));
        outputStreamE.write(outputBytes);

        inputStream.close();
        outputStreamE.close();
    }

    public static void decrypt(File inputFile, String outputFile) throws Exception {
        SecretKeySpec key = createKey();
        Cipher cipher = Cipher.getInstance(TRANSFORMATION, "SunJCE");
        byte[] iv = new byte[IV_SIZE];
        IvParameterSpec ivSpec = new IvParameterSpec(iv);
        cipher.init(Cipher.DECRYPT_MODE, key, ivSpec);

        FileInputStream inputStream = new FileInputStream(inputFile);
        byte[] inputBytes = new byte[(int) inputFile.length()];
        inputStream.read(inputBytes);

        byte[] outputBytes = cipher.doFinal(inputBytes);

        FileOutputStream outputStream = new FileOutputStream(outputFile);
        DataOutputStream outputStreamT = new DataOutputStream(new BufferedOutputStream(outputStream));
        outputStreamT.write(outputBytes);

        inputStream.close();
        outputStreamT.close();
    }

    private static SecretKeySpec createKey() throws Exception {
        if (secretKey == null) {
            // Generate a random byte array of the specified key size
            KeyGenerator keyGen = KeyGenerator.getInstance(ALGORITHM);
            keyGen.init(256);
            SecretKey key = keyGen.generateKey();
            byte[] keyBytes = key.getEncoded();

            // Create a SecretKeySpec object from the random byte array
            secretKey = new SecretKeySpec(keyBytes, ALGORITHM);

        }
        return secretKey;
    }

}