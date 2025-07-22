using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

class Program
{
    const int EntrySize = 16;
    const int KeySize = 5;
    const int ValueSize = 5;
    const int TableSize = 1 << 30; // 1 GB
    const int EntryCount = TableSize / EntrySize;

    static void Main()
    {
        Console.WriteLine("Generating table...");
        string path = "table.bin";
        if (!File.Exists(path))
            GenerateBinaryTable(path);

        Console.WriteLine("Loading table into memory...");
        byte[] data = File.ReadAllBytes(path);
        Span<byte> table = data;

        Console.WriteLine("Benchmarking AVX2 (batch scan)...");
        BenchmarkAVX2(table);

        Console.WriteLine("Benchmarking Span + BinarySearch...");
        BenchmarkSpanBinarySearch(table);

        Console.WriteLine("Benchmarking Dictionary...");
        BenchmarkDictionary(table);
    }

    static void BenchmarkAVX2(Span<byte> table)
    {
        var sw = Stopwatch.StartNew();
        Random rand = new Random(12345);
        int found = 0;
        Span<ulong> keys = stackalloc ulong[8];
        for (int t = 0; t < 10000; t += 8)
        {
            for (int i = 0; i < 8; i++)
                keys[i] = (ulong)rand.NextInt64() & 0xFFFFFFFFFF;

            if (Avx2.IsSupported)
                found += Avx2BatchLookup(table, keys);
        }
        sw.Stop();
        Console.WriteLine($"AVX2 lookup time: {sw.ElapsedMilliseconds} ms, found: {found}");
    }

    static int Avx2BatchLookup(Span<byte> table, Span<ulong> targets)
    {
        int count = 0;
        unsafe
        {
            fixed (byte* p = table)
            {
                for (int i = 0; i < EntryCount; i++)
                {
                    byte* entry = p + i * EntrySize + KeySize;
                    ulong value = Read5Byte(entry);
                    for (int j = 0; j < targets.Length; j++)
                        if (value == targets[j])
                            count++;
                }
            }
        }
        return count;
    }

    static void BenchmarkSpanBinarySearch(Span<byte> table)
    {
        var sw = Stopwatch.StartNew();
        Random rand = new Random(42);
        int found = 0;
        for (int i = 0; i < 10000; i++)
        {
            ulong key = (ulong)rand.NextInt64() & 0xFFFFFFFFFF;
            if (BinarySearchSpan(table, key) >= 0)
                found++;
        }
        sw.Stop();
        Console.WriteLine($"Span binary search: {sw.ElapsedMilliseconds} ms, found: {found}");
    }

    static void BenchmarkDictionary(Span<byte> table)
    {
        var sw = Stopwatch.StartNew();
        var dict = new Dictionary<ulong, byte[]>();
        for (int i = 0; i < EntryCount; i++)
        {
            int offset = i * EntrySize;
            ulong key = Read5Byte(table.Slice(offset, KeySize));
            byte[] value = table.Slice(offset + KeySize, ValueSize).ToArray();
            dict[key] = value;
        }

        int found = 0;
        Random rand = new Random(999);
        for (int i = 0; i < 10000; i++)
        {
            ulong key = (ulong)rand.NextInt64() & 0xFFFFFFFFFF;
            if (dict.ContainsKey(key))
                found++;
        }
        sw.Stop();
        Console.WriteLine($"Dictionary lookup: {sw.ElapsedMilliseconds} ms, found: {found}");
    }

    static int BinarySearchSpan(Span<byte> table, ulong target)
    {
        int left = 0, right = EntryCount - 1;
        while (left <= right)
        {
            int mid = (left + right) / 2;
            int offset = mid * EntrySize + KeySize;
            ulong value = Read5Byte(table.Slice(offset, ValueSize));
            if (value == target) return mid;
            else if (value < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong Read5Byte(ReadOnlySpan<byte> span)
    {
        ulong value = 0;
        for (int i = 0; i < 5; i++)
            value |= ((ulong)span[i]) << (i * 8);
        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    unsafe static ulong Read5Byte(byte* ptr)
    {
        ulong value = 0;
        for (int i = 0; i < 5; i++)
            value |= ((ulong)*(ptr + i)) << (i * 8);
        return value;
    }

    static void GenerateBinaryTable(string path)
    {
        Random rand = new Random(1234);
        byte[] buffer = new byte[TableSize];
        Span<byte> span = buffer;

        for (int i = 0; i < EntryCount; i++)
        {
            int offset = i * EntrySize;
            rand.NextBytes(span.Slice(offset, KeySize));
            rand.NextBytes(span.Slice(offset + KeySize, ValueSize));
        }

        span.Slice(0, EntryCount * EntrySize).Sort((a, b) =>
        {
            ulong va = Read5Byte(a.Slice(KeySize, ValueSize));
            ulong vb = Read5Byte(b.Slice(KeySize, ValueSize));
            return va.CompareTo(vb);
        }, EntrySize);

        File.WriteAllBytes(path, buffer);
    }
}