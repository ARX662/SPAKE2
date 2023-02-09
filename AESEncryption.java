import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.io.*;

public class AESEncryption {

  private static final String ALGORITHM = "AES";
  private static final String TRANSFORMATION = "AES/CBC/PKCS5Padding";
  private static final int IV_SIZE = 16; // bits
  public static void encrypt(byte[] password, File inputFile) throws Exception {
    SecretKeySpec key = createKey(password);
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

  public static void decrypt(byte[] password, File inputFile) throws Exception {
    SecretKeySpec key = createKey(password);
    Cipher cipher = Cipher.getInstance(TRANSFORMATION, "SunJCE");
    byte[] iv = new byte[IV_SIZE];
    IvParameterSpec ivSpec = new IvParameterSpec(iv);
    cipher.init(Cipher.DECRYPT_MODE, key, ivSpec);

    FileInputStream inputStream = new FileInputStream(inputFile);
    byte[] inputBytes = new byte[(int) inputFile.length()];
    inputStream.read(inputBytes);

    byte[] outputBytes = cipher.doFinal(inputBytes);

    FileOutputStream outputStream = new FileOutputStream(inputFile);
    DataOutputStream outputStreamT = new DataOutputStream(new BufferedOutputStream(outputStream));
    outputStreamT.write(outputBytes);

    inputStream.close();
    outputStreamT.close();
  }

  private static SecretKeySpec createKey(byte[] password) throws Exception {
    SecretKeySpec secretKey = new SecretKeySpec(password, ALGORITHM);

      return secretKey;

  }
}
