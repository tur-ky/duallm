using System;
using System.IO;
using System.Text;

namespace DualLlmPortal
{
    /// <summary>
    /// The few GGUF header fields the portal needs, read without loading the model.
    ///
    /// Only the key/value block at the front of the file is parsed, and the walk stops as soon as the
    /// wanted keys have been seen. That matters: the tokenizer arrays further in hold hundreds of
    /// thousands of strings, and the keys here all sit ahead of them, so a normal file is answered
    /// after a few kilobytes.
    /// </summary>
    public sealed class GgufMetadata
    {
        public string Architecture { get; private set; } = "";

        /// <summary>
        /// The context length the model was trained for, or 0 when it could not be read. llama.cpp
        /// will run past this, but quality degrades, so it is the natural ceiling for the offered list.
        /// </summary>
        public uint TrainingContext { get; private set; }

        private const uint Magic = 0x46554747; // "GGUF", little-endian

        // GGUF value type tags, from ggml/src/gguf.cpp.
        private const uint TypeUint8 = 0, TypeInt8 = 1, TypeUint16 = 2, TypeInt16 = 3;
        private const uint TypeUint32 = 4, TypeInt32 = 5, TypeFloat32 = 6, TypeBool = 7;
        private const uint TypeString = 8, TypeArray = 9, TypeUint64 = 10, TypeInt64 = 11;
        private const uint TypeFloat64 = 12;

        /// <summary>
        /// Reads what it can and returns null only when the file is not a readable GGUF at all.
        /// A file whose keys are missing comes back with the corresponding fields left at their
        /// defaults rather than throwing -- every caller has a sane fallback for that.
        /// </summary>
        public static GgufMetadata? TryRead(string path)
        {
            try
            {
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite,
                                                  64 * 1024, FileOptions.SequentialScan);
                using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

                if (reader.ReadUInt32() != Magic)
                {
                    return null;
                }

                var version = reader.ReadUInt32();
                if (version < 2 || version > 3)
                {
                    // v1 used 32-bit lengths throughout. Nothing in this collection is v1, and guessing
                    // at a layout that might not match would produce silently wrong numbers.
                    return null;
                }

                reader.ReadUInt64();                  // tensor count
                var kvCount = reader.ReadUInt64();

                var result = new GgufMetadata();
                string? wantedContextKey = null;

                for (ulong i = 0; i < kvCount; i++)
                {
                    var key = ReadString(reader);
                    var type = reader.ReadUInt32();

                    if (key == "general.architecture" && type == TypeString)
                    {
                        result.Architecture = ReadString(reader);
                        wantedContextKey = $"{result.Architecture}.context_length";
                    }
                    else if (wantedContextKey != null && key == wantedContextKey)
                    {
                        result.TrainingContext = ReadUnsignedScalar(reader, type);
                        return result;    // both fields in hand; the rest of the header is not needed
                    }
                    else
                    {
                        SkipValue(reader, type);
                    }
                }

                return result;
            }
            catch
            {
                // A truncated or unexpected file is a "no answer", never a crash: the caller falls
                // back to letting the user type a context size by hand.
                return null;
            }
        }

        private static string ReadString(BinaryReader reader)
        {
            var length = reader.ReadUInt64();
            if (length > 64 * 1024)
            {
                // No legitimate metadata string is this long; treat it as a corrupt header rather
                // than trying to allocate whatever the file claims.
                throw new InvalidDataException("implausible GGUF string length");
            }

            return Encoding.UTF8.GetString(reader.ReadBytes((int)length));
        }

        /// <summary>Reads any of the integer types as a uint, since that is all the callers want.</summary>
        private static uint ReadUnsignedScalar(BinaryReader reader, uint type) => type switch
        {
            TypeUint8 => reader.ReadByte(),
            TypeInt8 => (uint)reader.ReadSByte(),
            TypeUint16 => reader.ReadUInt16(),
            TypeInt16 => (uint)reader.ReadInt16(),
            TypeUint32 => reader.ReadUInt32(),
            TypeInt32 => (uint)reader.ReadInt32(),
            TypeUint64 => (uint)reader.ReadUInt64(),
            TypeInt64 => (uint)reader.ReadInt64(),
            _ => throw new InvalidDataException($"unexpected GGUF type {type} for a length field"),
        };

        private static void SkipValue(BinaryReader reader, uint type)
        {
            if (type == TypeArray)
            {
                var elementType = reader.ReadUInt32();
                var count = reader.ReadUInt64();

                if (elementType == TypeString)
                {
                    // Strings are variable length, so each one has to be stepped over individually.
                    // This is the tokenizer vocabulary in practice; the early-exit above normally
                    // means it is never reached.
                    for (ulong i = 0; i < count; i++)
                    {
                        var length = reader.ReadUInt64();
                        Seek(reader, (long)length);
                    }
                    return;
                }

                if (elementType == TypeArray)
                {
                    // Nested arrays have no fixed stride, so there is nothing to seek past.
                    throw new InvalidDataException("nested GGUF arrays are not supported");
                }

                Seek(reader, (long)count * FixedSize(elementType));
                return;
            }

            if (type == TypeString)
            {
                var length = reader.ReadUInt64();
                Seek(reader, (long)length);
                return;
            }

            Seek(reader, FixedSize(type));
        }

        private static int FixedSize(uint type) => type switch
        {
            TypeUint8 or TypeInt8 or TypeBool => 1,
            TypeUint16 or TypeInt16 => 2,
            TypeUint32 or TypeInt32 or TypeFloat32 => 4,
            TypeUint64 or TypeInt64 or TypeFloat64 => 8,
            _ => throw new InvalidDataException($"unknown GGUF type {type}"),
        };

        private static void Seek(BinaryReader reader, long bytes)
            => reader.BaseStream.Seek(bytes, SeekOrigin.Current);
    }
}
